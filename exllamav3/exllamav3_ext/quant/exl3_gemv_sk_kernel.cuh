#pragma once

// Split-k variant of the small-m EXL3 GEMV (exl3_gemv_kernel.cuh) for sm_75.
//
// The block-per-column-group GEMV leaves a partial last wave whenever the number of column groups is not a
// multiple of the co-resident block count (n = 5120 at 32 columns per group: 160 groups on 136 slots, so 24
// blocks run a second wave alone), and one-wave shapes with few groups (n = 1024) leave most SMs idle. Here the
// work space (column group x k-slice, group-major) is cut into equal contiguous ranges, one per block, and each
// block's range into equal contiguous ranges, one per warp, so every SM gets the same work for any shape.
//
// Each warp keeps an fp32 partial of the first and the last group its range touches in shared memory (groups in
// between are whole and go straight to the workspace), the block sums them per group in warp order and writes one
// partial per (block, group) to the workspace, and after a grid sync the output stage sums each column's block partials in
// block order (both deterministic) and applies the output scales and Hadamard transform. Summing inside the
// block first keeps the global partials at about two per column.
//
// Inner loop as in exl3_gemv_kernel: B streams to registers through a prefetch ring (ld.global.cs), tile words
// are resolved by lane shuffles, one m16n8k16 (two m16n8k8 on sm_75) MMA pair per 16x16 tile with fp16
// accumulation folded to fp32 once per ring cycle. The A fragments are loaded one k-slice ahead (every warp reads
// the same activations, from L2), and the output transform runs in registers on the reduced sums.
//
// 5-8 bpw (typically the output head) go through the same kernel with the tile words staged in shared memory.
//
// Measured on an RTX 2080 Ti with Qwen3.8-27B 4 bpw (6 bpw head), per-token weight time over every linear:
// 30.4 -> 26.6 ms at m = 1, 32.1 -> 30.1 ms at m = 8; 4 bpw n > 12288 stays on the wide GEMV config, which is a
// little faster there.
//
// Same kernel signature as exl3_gemm_kernel. The locks argument carries the fp32 workspace instead (graph
// parameter patching leaves it untouched): gridDim.x x max_groups_per_block x ROWS x COLS floats.

#include <cooperative_groups.h>
#include "exl3_gemv_kernel.cuh"

#define EXL3_GEMV_SK_THREADS 256
#define EXL3_GEMV_SK_WARPS (EXL3_GEMV_SK_THREADS / 32)
#define EXL3_GEMV_SK_WNT 4                // 16x16 tiles per warp (64 columns per group)
#define EXL3_GEMV_SK_PF 2                 // prefetch ring depth (k-slices)

template <int bits, bool c_fp32, int cb, int MMODE, int WNT, int PF>
__global__ __launch_bounds__(EXL3_GEMV_SK_THREADS)
void exl3_gemv_sk_kernel(EXL3_GEMM_ARGS)
{
    // Dispatched on sm_75 only (exl3_gemv_try_launch); other targets carry no code for it
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ != 750)
    __trap();
#else
    static_assert(bits >= 2 && bits <= 8, "exl3_gemv_sk_kernel supports 2 to 8 bpw");
    static_assert(8 % PF == 0, "prefetch depth must divide 8");
    constexpr int ROWS = MMODE == 0 ? 1 : EXL3_GEMV_MAX_M;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = 8 * bits;                                          // uint32 per 16x16 tile
    // 2-4 bpw: tile words resolved by lane shuffles. 5-8 bpw: a tile is 40-64 words, two warp loads, staged
    // through warp-private shared memory for the generic dq_dispatch
    constexpr bool BIG = bits > 4;
    constexpr bool TWO_PER_LOAD = bits == 2;                                  // two tiles per warp load
    constexpr int LOADS = BIG ? 2 * WNT : (TWO_PER_LOAD ? WNT / 2 : WNT);    // warp loads per k-slice
    constexpr int LSTRIDE = TWO_PER_LOAD ? 2 * TWORDS : (TWORDS < 32 ? TWORDS : 32);
    static_assert(!TWO_PER_LOAD || WNT % 2 == 0, "two tiles per warp load needs an even tile count per warp");

    auto grid = cooperative_groups::this_grid();
    float* __restrict__ ws = (float*) locks;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int num_warps = gridDim.x * EXL3_GEMV_SK_WARPS;
    const int gw = blockIdx.x * EXL3_GEMV_SK_WARPS + warp;

    const int ntiles = size_n / 16;
    const int kslices = size_k / 16;
    const int total = (size_n / COLS) * kslices;
    // Block and warp ranges are multiples of PF slices, so with group boundaries at multiples of 8 slices every
    // segment is a whole number of ring cycles
    const int span_w = CEIL_DIVIDE(CEIL_DIVIDE(total, num_warps), PF) * PF;
    const int span_b = span_w * EXL3_GEMV_SK_WARPS;
    const int max_gpb = CEIL_DIVIDE(span_b, kslices) + 1;                    // groups a block range can touch

    const uint32_t* B32 = (const uint32_t*) B;
    const size_t slice_stride = (size_t) ntiles * TWORDS;

    // Segment geometry and B prefetch ring
    int s = min(total, gw * span_w);
    const int e = min(total, s + span_w);
    int group = 0, ks0 = 0, myn = 0;
    const uint32_t* bp = nullptr;
    auto set_seg = [&] ()
    {
        group = s / kslices;
        ks0 = s - group * kslices;
        myn = min(e, (group + 1) * kslices) - s;
        s += myn;
        bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;
    };
    auto ld_b = [&] (int i, int l) -> uint32_t
    {
        if constexpr (BIG)
        {
            const int t = l >> 1, p = l & 1;
            return p == 0 || lane < TWORDS - 32 ? __ldcs(bp + (size_t) i * slice_stride + t * TWORDS + p * 32) : 0;
        }
        else if constexpr (LSTRIDE < 32)
            return lane < LSTRIDE ? __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE) : 0;
        else
            return __ldcs(bp + (size_t) i * slice_stride + l * LSTRIDE);
    };
    uint32_t pf[PF][LOADS];
    auto prologue = [&] ()
    {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                pf[d][l] = ld_b(d, l);
    };

    // Input scales and Hadamard transform, same as exl3_gemm_kernel
    {
        int total_warps = size_m * size_k / 128;
        for (int w = gw; w < total_warps; w += num_warps)
            had_hf_r_128_inner<true, false>
            (
                A + w * 128,
                A_had + w * 128,
                suh + (w * 128) % size_k,
                0.088388347648f  // 1/sqrt(128)
            );
        grid.sync();
        A = A_had;
    }

    const half2* A2 = (const half2*) A;
    const half2 hzero = __half2half2(__ushort_as_half(0));

    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t) r0 * (size_k / 2);
    const bool r0_ok = MMODE == 0 ? lane < 4 : r0 < size_m;

    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2)
    {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    else if constexpr (bits == 3)
    {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    // Per-warp partials of the first and the last segment, [warp][first/last][row][col]; segments in between cover
    // a whole group each and go straight to the block's workspace slot
    __shared__ float sh_part[EXL3_GEMV_SK_WARPS][2][ROWS][COLS];
    [[maybe_unused]] __shared__ uint32_t sh_stage[BIG ? EXL3_GEMV_SK_WARPS : 1][BIG ? WNT * TWORDS : 1];
    const int g_lo_b = min(total, blockIdx.x * span_b) / kslices;

    for (int seg = 0; s < e; ++seg)
    {
        set_seg();
        prologue();

        half2 a_nx0 = hzero, a_nx1 = hzero;
        if (r0_ok)
        {
            const size_t a_col = (size_t) ks0 * 8 + (lane & 3);
            a_nx0 = A2[a_row0 + a_col];
            a_nx1 = A2[a_row0 + a_col + 4];
        }

        FragC_h ch[WNT][2] = {};
        float2 acc0[WNT][2] = {};

        for (int ib = 0; ib < myn; ib += PF)
        {
            #pragma unroll
            for (int d = 0; d < PF; ++d)
            {
                const int i = ib + d;

                uint32_t bw[LOADS];
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    bw[l] = pf[d][l];

                if (i + PF < myn)
                {
                    #pragma unroll
                    for (int l = 0; l < LOADS; ++l)
                        pf[d][l] = ld_b(i + PF, l);
                }

                FragB a01, a23;
                a01[0] = a_nx0;
                a23[0] = a_nx1;
                a01[1] = hzero;
                a23[1] = hzero;
                if (i + 1 < myn && r0_ok)
                {
                    const size_t a_col = (size_t) (ks0 + i + 1) * 8 + (lane & 3);
                    a_nx0 = A2[a_row0 + a_col];
                    a_nx1 = A2[a_row0 + a_col + 4];
                }

                if constexpr (BIG)
                {
                    __syncwarp();
                    #pragma unroll
                    for (int l = 0; l < LOADS; ++l)
                        if ((l & 1) == 0 || lane < TWORDS - 32)
                            sh_stage[warp][(l >> 1) * TWORDS + (l & 1) * 32 + lane] = bw[l];
                    __syncwarp();
                }

                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                {
                    FragB f0, f1;
                    if constexpr (BIG)
                    {
                        dq_dispatch<bits, cb>(&sh_stage[warp][t * TWORDS], lane << 3, f0, f1);
                    }
                    else if constexpr (bits == 4)
                    {
                        uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                        exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                    }
                    else if constexpr (bits == 2)
                    {
                        const uint32_t w = bw[t >> 1];
                        const int base = (t & 1) << 4;
                        uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                        uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                        exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                    }
                    else
                    {
                        uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                        uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                        exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                    }
                    exl3_gemv_ns::mma_ab_h(a01, a23, f0, ch[t][0]);
                    exl3_gemv_ns::mma_ab_h(a01, a23, f1, ch[t][1]);
                }

                if (d == PF - 1)
                {
                    #pragma unroll
                    for (int t = 0; t < WNT; ++t)
                        #pragma unroll
                        for (int f = 0; f < 2; ++f)
                        {
                            acc0[t][f].x += __low2float(ch[t][f][0]);
                            acc0[t][f].y += __high2float(ch[t][f][0]);
                            ch[t][f][0] = hzero;
                        }
                }
            }
        }

        // Partial of this segment: lane l holds row l/4, cols tile * 16 + frag * 8 + 2 * (l % 4) (+1)
        if (MMODE == 0 ? lane < 4 : r0 < ROWS)
        {
            const int sr = MMODE == 0 ? 0 : r0;
            float* sp = seg == 0 ? &sh_part[warp][0][sr][2 * (lane & 3)] :
                        s >= e   ? &sh_part[warp][1][sr][2 * (lane & 3)] :
                                   ws + (((size_t) blockIdx.x * max_gpb + group - g_lo_b) * ROWS + sr) * COLS + 2 * (lane & 3);
            #pragma unroll
            for (int t = 0; t < WNT; ++t)
                #pragma unroll
                for (int f = 0; f < 2; ++f)
                    *((float2*) (sp + t * 16 + f * 8)) = acc0[t][f];
        }
    }
    __syncthreads();

    // Block partial per group touched by the block range, summed over the warps that touched it in warp order
    const int rows_out = MMODE == 0 ? 1 : size_m;
    {
        const int bs = min(total, blockIdx.x * span_b);
        const int be = min(total, bs + span_b);
        if (bs < be)
        {
            const int g_lo = bs / kslices;
            const int g_hi = (be - 1) / kslices;
            for (int idx = threadIdx.x; idx < (g_hi - g_lo + 1) * rows_out * COLS; idx += EXL3_GEMV_SK_THREADS)
            {
                const int j = idx / (rows_out * COLS);
                const int rc = idx - j * rows_out * COLS;
                const int r = rc / COLS;
                const int c = rc - r * COLS;
                const int g = g_lo + j;
                float sum = 0.0f;
                bool direct = false;
                #pragma unroll
                for (int w = 0; w < EXL3_GEMV_SK_WARPS; ++w)
                {
                    const int ws0 = bs + w * span_w;
                    const int we = min(be, ws0 + span_w);
                    if (ws0 >= we) continue;
                    const int fg = ws0 / kslices;
                    const int lg = (we - 1) / kslices;
                    if (g > fg && g < lg) direct = true;
                    else if (g == fg) sum += sh_part[w][0][r][c];
                    else if (g == lg) sum += sh_part[w][1][r][c];
                }
                // A group inside one warp's range was written by that warp and has no other contributor
                if (!direct) ws[(((size_t) blockIdx.x * max_gpb + j) * ROWS + r) * COLS + c] = sum;
            }
        }
    }

    grid.sync();

    // Output: each column's block partials in block order, then the output scales and Hadamard transform in
    // registers, same operations as had_ff_r_128_inner / had_hf_r_128_inner (lane holds columns 4 * lane .. + 3)
    const int chunks_row = size_n / 128;
    for (int w = gw; w < rows_out * chunks_row; w += num_warps)
    {
        const int r = w / chunks_row;
        const int c0 = (w - r * chunks_row) * 128;
        const int col = c0 + 4 * lane;
        const int g = col / COLS;
        const int c = col - g * COLS;
        const int b_lo = g * kslices / span_b;
        const int nb = ((g + 1) * kslices - 1) / span_b - b_lo + 1;
        const half4 sc = ((const half4*) (svh + c0))[lane];
        auto part = [&] (int u) -> float4
        {
            const int b = b_lo + u;
            return __ldcg((const float4*) (ws + (((size_t) b * max_gpb + g - b * span_b / kslices) * ROWS + r) * COLS + c));
        };
        float4 pv[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) pv[u] = u < nb ? part(u) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 v = pv[0];
        #pragma unroll
        for (int u = 1; u < 4; ++u) { v.x += pv[u].x; v.y += pv[u].y; v.z += pv[u].z; v.w += pv[u].w; }
        for (int u = 4; u < nb; ++u) { float4 q = part(u); v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w; }

        const float r_scale = 0.088388347648f;  // 1/sqrt(128)
        if constexpr (c_fp32)
        {
            float s0 = v.x + v.y, d0 = v.x - v.y, s1 = v.z + v.w, d1 = v.z - v.w;
            v.x = s0 + s1; v.y = d0 + d1; v.z = s0 - s1; v.w = d0 - d1;
            shuffle_had_f2x32(v.x, v.y, lane);
            shuffle_had_f2x32(v.z, v.w, lane);
            v.x *= r_scale; v.y *= r_scale; v.z *= r_scale; v.w *= r_scale;
            v.x *= __low2float(sc.x); v.y *= __high2float(sc.x); v.z *= __low2float(sc.y); v.w *= __high2float(sc.y);
            ((float4*) C)[((size_t) r * size_n + c0) / 4 + lane] = v;
        }
        else
        {
            // C is fp16 on this path: the sums round to half before the transform, as they do through memory
            const half2 hx = __floats2half2_rn(v.x, v.y), hy = __floats2half2_rn(v.z, v.w);
            const float v0 = __low2float(hx), v1 = __high2float(hx), v2 = __low2float(hy), v3 = __high2float(hy);
            float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
            float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
            shuffle_had_f4x32(h0, h1, h2, h3, lane);
            half4 o;
            o.x = __hmul2(__floats2half2_rn(h0 * r_scale, h1 * r_scale), sc.x);
            o.y = __hmul2(__floats2half2_rn(h2 * r_scale, h3 * r_scale), sc.y);
            ((half4*) C)[((size_t) r * size_n + c0) / 4 + lane] = o;
        }
    }
#endif
}
