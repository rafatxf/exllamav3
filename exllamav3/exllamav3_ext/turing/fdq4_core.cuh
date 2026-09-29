#pragma once

// fdq4: flash-decoding attention straight from ExLlamaV3's packed 4-bit K/V cache, tuned for
// Turing (sm_75) and GQA models with head_dim 256 (Qwen3.5/3.8: 24 q heads, 4 kv heads). The graph-path
// cubins are compiled per shape and also cover head_dim 512 (FD_HD = 512: 8 warps of 64 dims) and
// sliding-window layers (Gemma 4: 1024-token windows, 512-dim global layers).
//
// Why: on sm_75 the Triton paged decode kernel reaches 17-50 GB/s at long context (tile ladders
// shrunk for 64 KB smem, 16-row q blocks for 6-head groups, KV re-read per head block). This kernel
// reads each packed K/V byte once per (kv head, split), for all q rows of the GQA group and all
// draft positions at once.
//
// Cache layout (exllamav3 CacheLayer_quant, 4 bits): per token row, n_kv_heads * head_dim / 32 groups
// of 32 values; each group is 4 consecutive uint32 words (8 nibbles each: value v of the group in word
// v/8, nibble v%8) plus one fp16 scale. value = (nibble - 7.5) * scale / 8, in a domain rotated per
// 32-group by the normalized Sylvester Hadamard matrix H32. H32 is symmetric orthonormal, so
// q.k = (H q).k_rot and o = H (sum p v_rot): q is rotated once on load, o once in the combine pass.
//
// Work split: grid (splits, n_kv_heads, bsz), 4 warps per block, chunks of 16 tokens. Warp w owns
// head dims [64w, 64w+64):
//   S^T(16 tok x R) partial over its 64 dims = K(16 x 64) . Q^T(64 x R)      mma m16n8k8, M = tokens
//   cross-warp reduce + online softmax, TPR threads per row (2 barriers per chunk)
//   O^T(64 dims x R) = alpha * O^T + V^T(64 x 16) . P^T(16 x R)             mma m16n8k8, M = dims
// Dequantization uses the fp16 magic-number trick ((nib | 0x6400) - 1024 = nib, exact), so dims are
// permuted inside the fragments to whatever pairs the bit tricks produce cheaply:
//   S : k-step s = 4g + p, k-index 2tig+e  ->  word 4g+tig, nibble p + 4e      (pairs (p, p+4))
//       q is stored in smem with every 8-dim block reordered (0,4,1,5,2,6,3,7) to match
//   PV: dim-row 16mt + gid + 8h            ->  word gid,   nibble 2mt + h      (byte_perm of 2 tokens)
// The next chunk's packed words are loaded into registers while the current chunk computes.
// A combine kernel merges the per-split (m, l, o) and applies the output rotation.

#pragma once
#include <cuda_fp16.h>
#include <stdint.h>
#include <algorithm>

#ifndef FD_HD
#define FD_HD 256
#endif
#define FD_WARPS (FD_HD / 64)   // warp w owns head dims [64w, 64w + 64)
#define FD_THREADS (FD_WARPS * 32)
#define FD_CHUNK 16
#define FD_PAGE 256
#define FD_MAXR 48          // max q rows per kv head (q_len * group): static smem budget

static __device__ __forceinline__ void mma_1688(float* c, uint32_t a0, uint32_t a1, uint32_t b0)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

static __device__ __forceinline__ uint32_t lop3_and_or(uint32_t a, uint32_t b, uint32_t c)   // (a & b) | c
{
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// (magic half2 holding 1024 + n) -> (n - 7.5) * sc per lane with a single rounding:
// (1024 + n) - 1031 = n - 7 is exact, and bias = -0.5 * sc is exact (power-of-two multiple), so the
// fused (n - 7) * sc - 0.5 * sc rounds once and carries no systematic per-group bias
static __device__ __forceinline__ uint32_t deq2(uint32_t magic, __half2 sc, __half2 bias)
{
    const __half2 k1031 = __float2half2_rn(1031.0f);
    __half2 n = __hsub2(*reinterpret_cast<__half2*>(&magic), k1031);
    __half2 r = __hfma2(n, sc, bias);
    return *reinterpret_cast<uint32_t*>(&r);
}

static __device__ __forceinline__ float hsign(int i, int j) { return (__popc(i & j) & 1) ? -1.0f : 1.0f; }

// Softmax threads per q row: the largest power of two <= 16 with RP * TPR <= min(FD_THREADS, 128). The same values
// as before for 128- and 256-thread blocks (head_dim 256 / 512); 64-thread blocks (head_dim 128) get fewer
template <int RP>
static __device__ __forceinline__ constexpr int fd_tpr()
{
    constexpr int T = (FD_THREADS < 128 ? FD_THREADS : 128) / RP;
    return T >= 16 ? 16 : T >= 8 ? 8 : T >= 4 ? 4 : T >= 2 ? 2 : 1;
}

template <int NT>   // NT = number of 8-row n-tiles covering R rows
static __device__ __forceinline__ void fdq4_split_body(
    const int split, const int kvh, const int b, const int splits,
    const half* __restrict__ q,            // (bsz, ql, nq, HD)
    const uint32_t* __restrict__ qk,       // (pages, PAGE, row_words)
    const half* __restrict__ sk,           // (pages, PAGE, row_groups)
    const uint32_t* __restrict__ qv,
    const half* __restrict__ sv,
    const int* __restrict__ block_table,   // (bsz, pps)
    const int* __restrict__ cache_seqlens, // (bsz)
    float* __restrict__ part_o,            // (bsz, nkv, splits, R, HD)
    float* __restrict__ part_ml,           // (bsz, nkv, splits, R, 2)
    int ql, int nq, int nkv, int pps, int split_len, int pre_appended, float scale_log2,
    const int row0, const int nrows,       // this block's q rows [row0, row0 + nrows) of the R = ql * G
    const int win_left = -1,               // sliding window: keys >= query position - win_left (-1: none)
    const int noncausal = 0)               // every row sees every key (draft blocks attending to themselves)
{
    constexpr int RP = NT * 8;
    constexpr int TPR = fd_tpr<RP>();   // softmax threads per row
    constexpr int TOK_PER = FD_CHUNK / TPR;
    // Row stride of the partial-score buffer: the softmax threads of a warp read (token sub + i TPR, row r) for
    // 32 / TPR rows and TPR subs; a stride of 32 / TPR (mod 32) puts those 32 reads in distinct banks (RP + 1
    // left them up to 4-way conflicted, and the kernel was bound by shared-memory wavefronts at head_dim 512)
    constexpr int SP_STRIDE = RP >= 16 ? RP + ((32 / TPR - RP) % 32 + 32) % 32 : RP + 1;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int G = nq / nkv, Rt = ql * G, R = nrows;   // Rt: rows in the partial layout, R: rows of this block
    const int row_words = nkv * FD_HD / 8, row_groups = nkv * FD_HD / 32;
    const int seqlen = cache_seqlens[b];
    const int kv_len = seqlen + pre_appended;
    // Sliding-window layers split only [first query position - window, kv_len) across the splits (same ranges as
    // the Triton decode kernel); keys below it carry zero weight
    int t0, t1;
    if (win_left >= 0)
    {
        const int w_lo = max(kv_len - ql - win_left, 0) / FD_CHUNK * FD_CHUNK;
        const int w_span = (((kv_len - w_lo) + splits - 1) / splits + FD_CHUNK - 1) / FD_CHUNK * FD_CHUNK;
        t0 = w_lo + split * w_span;
        t1 = min(t0 + w_span, kv_len);
    }
    else
    {
        t0 = split * split_len;
        t1 = min(t0 + split_len, kv_len);
    }

    // One static buffer, two phases: during setup it holds 8 q rows at a time (rotated, scaled,
    // fragment-ordered) while each warp lifts them into registers; afterwards the same bytes hold the
    // per-chunk partial scores, probabilities and rescale factors. Keeps NT = 6 at ~14 KB of smem.
    constexpr int SP_BYTES = FD_WARPS * FD_CHUNK * SP_STRIDE * 4;
    // P, row-major per q row with the chunk's tokens contiguous (padded), so a B fragment (tokens 2 tig, 2 tig + 1
    // of row gid) is one 32-bit load
    constexpr int PS_STRIDE = FD_CHUNK + 2;
    constexpr int PS_BYTES = RP * PS_STRIDE * 2;
    constexpr int AL_BYTES = RP * 4;
    constexpr int LOOP_BYTES = SP_BYTES + PS_BYTES + AL_BYTES;
    constexpr int QS_BYTES = 8 * FD_HD * 2;
    __shared__ __align__(16) unsigned char smem_raw[LOOP_BYTES > QS_BYTES ? LOOP_BYTES : QS_BYTES];
    half* qs = reinterpret_cast<half*>(smem_raw);
    auto sp = reinterpret_cast<float (*)[FD_CHUNK][SP_STRIDE]>(smem_raw);
    auto ps = reinterpret_cast<half (*)[PS_STRIDE]>(smem_raw + SP_BYTES);
    float* alpha_s = reinterpret_cast<float*>(smem_raw + SP_BYTES + PS_BYTES);

    const size_t pbase = (((size_t)b * nkv + kvh) * splits + split) * Rt + row0;

    if (t0 >= t1)
    {
        for (int r = tid; r < R; r += FD_THREADS)
        {
            part_ml[(pbase + r) * 2 + 0] = -INFINITY;
            part_ml[(pbase + r) * 2 + 1] = 0.0f;
        }
        return;
    }

    const int* bt = block_table + (size_t)b * pps;
    const int kw = kvh * (FD_HD / 8) + warp * 8;     // this warp's first word within a token row
    const int kg = kvh * (FD_HD / 32) + warp * 2;    // this warp's first scale group

    // ---- register double buffer of packed K/V for one chunk
    // K: tokens gid, gid+8 ; words tig (group kg), 4+tig (group kg+1)
    // V: tokens 2tig+e (+8ks) ; word gid (group kg + gid/4)
    uint32_t kwd[2][2], vwd[2][2];
    half ksc[2][2], vsc[2][2];
    // Per-thread element offsets from a chunk's first row (fixed), and the chunk's first row pointer, which only
    // needs the block table when the chunk enters a new page (a chunk never straddles a page)
    const int k_off0 = gid * row_words + kw + tig, k_off1 = k_off0 + 8 * row_words;
    const int ks_off0 = gid * row_groups + kg, ks_off1 = ks_off0 + 8 * row_groups;
    const int v_off = (2 * tig) * row_words + kw + gid;
    const int vs_off = (2 * tig) * row_groups + kg + (gid >> 2);
    int cur_page = -1;
    size_t page_row = 0;
    auto load_chunk = [&](int c, uint32_t (&kw_)[2][2], half (&ks_)[2][2], uint32_t (&vw_)[2][2], half (&vs_)[2][2])
    {
        const int pg = c / FD_PAGE;
        if (pg != cur_page) { cur_page = pg; page_row = (size_t)__ldg(bt + pg) * FD_PAGE; }
        const size_t row0c = page_row + (c % FD_PAGE);
        const uint32_t* kb = qk + row0c * row_words;
        const half* ksb = sk + row0c * row_groups;
        const uint32_t* vb = qv + row0c * row_words;
        const half* vsb = sv + row0c * row_groups;
        if (c + FD_CHUNK <= t1)
        {
            kw_[0][0] = __ldg(kb + k_off0); kw_[0][1] = __ldg(kb + k_off0 + 4);
            kw_[1][0] = __ldg(kb + k_off1); kw_[1][1] = __ldg(kb + k_off1 + 4);
            ks_[0][0] = __ldg(ksb + ks_off0); ks_[0][1] = __ldg(ksb + ks_off0 + 1);
            ks_[1][0] = __ldg(ksb + ks_off1); ks_[1][1] = __ldg(ksb + ks_off1 + 1);
            #pragma unroll
            for (int ks = 0; ks < 2; ++ks)
                #pragma unroll
                for (int e = 0; e < 2; ++e)
                {
                    vw_[ks][e] = __ldg(vb + v_off + (ks * 8 + e) * row_words);
                    vs_[ks][e] = __ldg(vsb + vs_off + (ks * 8 + e) * row_groups);
                }
            return;
        }
        // Last, partial chunk
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            int o = gid + 8 * h;
            if (c + o < t1)
            {
                const uint32_t* kr = kb + (h ? k_off1 : k_off0);
                kw_[h][0] = __ldg(kr);
                kw_[h][1] = __ldg(kr + 4);
                const half* sr = ksb + (h ? ks_off1 : ks_off0);
                ks_[h][0] = __ldg(sr);
                ks_[h][1] = __ldg(sr + 1);
            }
            else { kw_[h][0] = kw_[h][1] = 0; ks_[h][0] = ks_[h][1] = __float2half(0.0f); }
        }
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks)
            #pragma unroll
            for (int e = 0; e < 2; ++e)
            {
                int o = ks * 8 + 2 * tig + e;
                if (c + o < t1)
                {
                    vw_[ks][e] = __ldg(vb + v_off + (ks * 8 + e) * row_words);
                    vs_[ks][e] = __ldg(vsb + vs_off + (ks * 8 + e) * row_groups);
                }
                else { vw_[ks][e] = 0; vs_[ks][e] = __float2half(0.0f); }
            }
    };
    load_chunk(t0, kwd, ksc, vwd, vsc);   // in flight while q is prepared

    // ---- q rows of this kv head, 8 at a time: rotate by H32 per group, fold softmax scale * log2(e),
    // store every 8-dim block in (0,4,1,5,2,6,3,7) order, then lift this warp's 64-dim Q^T fragments
    // (k-step s = 4g + p: b0 = {q[word 4g+tig nib p], [.. nib p+4]}) into registers
    uint32_t qf[NT][8];
    const int wd = warp * 64;
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt)
    {
        // One 32-dim group per warp pass, lane = dim: fast Walsh-Hadamard transform (5 butterfly stages) gives
        // y_i = sum_j (-1)^popc(i & j) x_j, the Sylvester H32 product
        for (int grp = warp; grp < 8 * (FD_HD / 32); grp += FD_WARPS)
        {
            int rl = grp / (FD_HD / 32), d = (grp % (FD_HD / 32)) * 32 + lane, r = nt * 8 + rl;
            float acc = 0.0f;
            if (r < R)
            {
                int qi = (row0 + r) / G, hg = (row0 + r) % G;
                acc = __half2float(q[(((size_t)b * ql + qi) * nq + kvh * G + hg) * FD_HD + d]);
            }
            #pragma unroll
            for (int m = 1; m < 32; m <<= 1)
            {
                float other = __shfl_xor_sync(0xffffffffu, acc, m);
                acc = (lane & m) ? other - acc : acc + other;
            }
            acc *= 0.17677669529663687f * scale_log2;
            int o = d & 7;
            qs[rl * FD_HD + (d & ~7) + 2 * (o & 3) + (o >> 2)] = __float2half(acc);
        }
        __syncthreads();
        const half* qr = qs + gid * FD_HD + wd;
        #pragma unroll
        for (int s = 0; s < 8; ++s)
            qf[nt][s] = *reinterpret_cast<const uint32_t*>(qr + (4 * (s >> 2) + tig) * 8 + 2 * (s & 3));
        __syncthreads();
    }

    float oacc[4][NT][4];
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) oacc[mt][nt][i] = 0.0f;

    // softmax ownership: thread -> (row sm_row, tokens sm_sub + i*TPR)
    // (warp-uniform participation so the xor-shuffles below always see full warps)
    const int sm_row_raw = tid / TPR, sm_sub = tid % TPR;
    const bool sm_valid = tid < RP * TPR;
    const bool sm_active = warp * 32 < RP * TPR;
    const int sm_row = sm_valid ? sm_row_raw : RP - 1;
    const int sm_qi = (row0 + sm_row) / G;
    float m_run = -INFINITY, l_run = 0.0f;

    const __half2 m05 = __float2half2_rn(-0.5f);
    const half eighth = __float2half(0.125f);

    for (int c = t0; c < t1; c += FD_CHUNK)
    {
        uint32_t nkwd[2][2], nvwd[2][2];
        half nksc[2][2], nvsc[2][2];
        const bool has_next = c + FD_CHUNK < t1;
        if (has_next) load_chunk(c + FD_CHUNK, nkwd, nksc, nvwd, nvsc);

        // ---------------- S partial over this warp's 64 dims
        float sacc[NT][4];
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) sacc[nt][i] = 0.0f;
        #pragma unroll
        for (int g = 0; g < 2; ++g)
        {
            __half2 sc0 = __half2half2(__hmul(ksc[0][g], eighth));
            __half2 sc1 = __half2half2(__hmul(ksc[1][g], eighth));
            __half2 bi0 = __hmul2(sc0, m05);
            __half2 bi1 = __hmul2(sc1, m05);
            #pragma unroll
            for (int p = 0; p < 4; ++p)
            {
                uint32_t a0 = deq2(lop3_and_or(kwd[0][g] >> (4 * p), 0x000F000Fu, 0x64006400u), sc0, bi0);
                uint32_t a1 = deq2(lop3_and_or(kwd[1][g] >> (4 * p), 0x000F000Fu, 0x64006400u), sc1, bi1);
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) mma_1688(sacc[nt], a0, a1, qf[nt][4 * g + p]);
            }
        }
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
        {
            int col = nt * 8 + 2 * tig;
            sp[warp][gid][col] = sacc[nt][0];
            sp[warp][gid][col + 1] = sacc[nt][1];
            sp[warp][gid + 8][col] = sacc[nt][2];
            sp[warp][gid + 8][col + 1] = sacc[nt][3];
        }
        __syncthreads();

        // ---------------- online softmax: TPR threads per row, TOK_PER tokens each
        if (sm_active)
        {
            float sv_[TOK_PER];
            float mc = -INFINITY;
            const int limit = noncausal ? t1 : min(t1, seqlen + sm_qi + 1);   // causal
            const int lower = win_left >= 0 ? seqlen + sm_qi - win_left : 0;
            #pragma unroll
            for (int i = 0; i < TOK_PER; ++i)
            {
                int k = sm_sub + i * TPR;
                float s = 0.0f;
                #pragma unroll
                for (int w = 0; w < FD_WARPS; ++w) s += sp[w][k][sm_row];
                s = (sm_valid && sm_row < R && c + k < limit && c + k >= lower) ? s : -INFINITY;
                sv_[i] = s;
                mc = fmaxf(mc, s);
            }
            #pragma unroll
            for (int off = TPR / 2; off > 0; off >>= 1) mc = fmaxf(mc, __shfl_xor_sync(0xffffffffu, mc, off));
            float m_new = fmaxf(m_run, mc);
            float al = (m_new == -INFINITY) ? 1.0f : exp2f(m_run - m_new);
            float lsum = 0.0f;
            #pragma unroll
            for (int i = 0; i < TOK_PER; ++i)
            {
                float pv = (m_new == -INFINITY) ? 0.0f : exp2f(sv_[i] - m_new);
                lsum += pv;
                if (sm_valid) ps[sm_row][sm_sub + i * TPR] = __float2half(pv);
            }
            #pragma unroll
            for (int off = TPR / 2; off > 0; off >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
            l_run = l_run * al + lsum;
            m_run = m_new;
            if (sm_valid && sm_sub == 0) alpha_s[sm_row] = al;
        }
        __syncthreads();

        // ---------------- PV over this warp's 64 dims
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
        {
            float a0 = alpha_s[nt * 8 + 2 * tig], a1 = alpha_s[nt * 8 + 2 * tig + 1];
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt)
            {
                oacc[mt][nt][0] *= a0; oacc[mt][nt][1] *= a1;
                oacc[mt][nt][2] *= a0; oacc[mt][nt][3] *= a1;
            }
        }
        uint32_t pb[2][NT];
        __half2 sc[2], bi[2];
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks)
        {
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
            {
                pb[ks][nt] = *reinterpret_cast<const uint32_t*>(&ps[nt * 8 + gid][ks * 8 + 2 * tig]);
            }
            sc[ks] = __halves2half2(__hmul(vsc[ks][0], eighth), __hmul(vsc[ks][1], eighth));
            bi[ks] = __hmul2(sc[ks], m05);
        }
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt)
        {
            #pragma unroll
            for (int ks = 0; ks < 2; ++ks)
            {
                // byte mt of both tokens' words -> 16-bit lanes (tok0, tok1); lo nibble = dim-row gid, hi = gid+8
                uint32_t t = __byte_perm(vwd[ks][0], vwd[ks][1], 0x4400 + mt * 0x1111);
                uint32_t a0 = deq2(lop3_and_or(t, 0x000F000Fu, 0x64006400u), sc[ks], bi[ks]);
                uint32_t a1 = deq2(lop3_and_or(t >> 4, 0x000F000Fu, 0x64006400u), sc[ks], bi[ks]);
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) mma_1688(oacc[mt][nt], a0, a1, pb[ks][nt]);
            }
        }

        if (has_next)
        {
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                {
                    kwd[i][j] = nkwd[i][j]; ksc[i][j] = nksc[i][j];
                    vwd[i][j] = nvwd[i][j]; vsc[i][j] = nvsc[i][j];
                }
        }
    }

    // ---- store partials (rotated domain); accumulator dim-row 16mt + gid + 8h -> dim wd + gid*8 + 2mt + h
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt)
    {
        int r0 = nt * 8 + 2 * tig;
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt)
        {
            int d0 = wd + gid * 8 + 2 * mt;
            if (r0 < R)
                *reinterpret_cast<float2*>(&part_o[(pbase + r0) * FD_HD + d0]) = make_float2(oacc[mt][nt][0], oacc[mt][nt][2]);
            if (r0 + 1 < R)
                *reinterpret_cast<float2*>(&part_o[(pbase + r0 + 1) * FD_HD + d0]) = make_float2(oacc[mt][nt][1], oacc[mt][nt][3]);
        }
    }
    if (sm_valid && sm_sub == 0 && sm_row < R)
    {
        part_ml[(pbase + sm_row) * 2 + 0] = m_run;
        part_ml[(pbase + sm_row) * 2 + 1] = l_run;
    }
}


// ---- fd16: the same split/softmax/combine structure over fp16 K/V (unrotated, no dequantization), for fp16
// caches such as SlidingAttention's per-slot window ring, where the Triton decode kernel reaches ~46 GB/s on sm_75.
//   S : k-step s over this warp's 64 dims: thread (gid, tig) holds dims wd + 16 tig + 8 (s / 4) + 2 (s % 4) (+1)
//       of tokens gid and gid + 8, i.e. two contiguous 16-byte loads per token row; q is lifted in the same order
//   PV: dim-row gid <-> dim wd + 16 mt + 2 gid, row gid + 8 <-> that dim + 1: one 32-bit load per (token, mt) gives
//       both rows, and the accumulator pairs land on contiguous dims for the partial store
template <int NT>
static __device__ __forceinline__ void fd16_split_body(
    const int split, const int kvh, const int b, const int splits,
    const half* __restrict__ q,            // (bsz, ql, nq, HD)
    const half* __restrict__ kc,           // (pages, PAGE, nkv, HD)
    const half* __restrict__ vc,
    const int* __restrict__ block_table,   // (bsz, pps)
    const int* __restrict__ cache_seqlens, // (bsz)
    float* __restrict__ part_o,            // (bsz, nkv, splits, R, HD)
    float* __restrict__ part_ml,           // (bsz, nkv, splits, R, 2)
    int ql, int nq, int nkv, int pps, int split_len, int pre_appended, float scale_log2,
    const int row0, const int nrows, const int win_left, const int noncausal = 0)
{
    constexpr int RP = NT * 8;
    constexpr int TPR = fd_tpr<RP>();   // softmax threads per row
    constexpr int TOK_PER = FD_CHUNK / TPR;
    // Row stride of the partial-score buffer: the softmax threads of a warp read (token sub + i TPR, row r) for
    // 32 / TPR rows and TPR subs; a stride of 32 / TPR (mod 32) puts those 32 reads in distinct banks (RP + 1
    // left them up to 4-way conflicted, and the kernel was bound by shared-memory wavefronts at head_dim 512)
    constexpr int SP_STRIDE = RP >= 16 ? RP + ((32 / TPR - RP) % 32 + 32) % 32 : RP + 1;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int G = nq / nkv, Rt = ql * G, R = nrows;
    const int row_h = nkv * FD_HD;         // halves per token row
    const int seqlen = cache_seqlens[b];
    const int kv_len = seqlen + pre_appended;
    int t0, t1;
    if (win_left >= 0)
    {
        const int w_lo = max(kv_len - ql - win_left, 0) / FD_CHUNK * FD_CHUNK;
        const int w_span = (((kv_len - w_lo) + splits - 1) / splits + FD_CHUNK - 1) / FD_CHUNK * FD_CHUNK;
        t0 = w_lo + split * w_span;
        t1 = min(t0 + w_span, kv_len);
    }
    else
    {
        t0 = split * split_len;
        t1 = min(t0 + split_len, kv_len);
    }

    constexpr int SP_BYTES = FD_WARPS * FD_CHUNK * SP_STRIDE * 4;
    // P, row-major per q row with the chunk's tokens contiguous (padded), so a B fragment (tokens 2 tig, 2 tig + 1
    // of row gid) is one 32-bit load
    constexpr int PS_STRIDE = FD_CHUNK + 2;
    constexpr int PS_BYTES = RP * PS_STRIDE * 2;
    constexpr int AL_BYTES = RP * 4;
    constexpr int LOOP_BYTES = SP_BYTES + PS_BYTES + AL_BYTES;
    constexpr int QS_BYTES = 8 * FD_HD * 2;
    __shared__ __align__(16) unsigned char smem_raw[LOOP_BYTES > QS_BYTES ? LOOP_BYTES : QS_BYTES];
    half* qs = reinterpret_cast<half*>(smem_raw);
    auto sp = reinterpret_cast<float (*)[FD_CHUNK][SP_STRIDE]>(smem_raw);
    auto ps = reinterpret_cast<half (*)[PS_STRIDE]>(smem_raw + SP_BYTES);
    float* alpha_s = reinterpret_cast<float*>(smem_raw + SP_BYTES + PS_BYTES);

    const size_t pbase = (((size_t)b * nkv + kvh) * splits + split) * Rt + row0;

    if (t0 >= t1)
    {
        for (int r = tid; r < R; r += FD_THREADS)
        {
            part_ml[(pbase + r) * 2 + 0] = -INFINITY;
            part_ml[(pbase + r) * 2 + 1] = 0.0f;
        }
        return;
    }

    const int* bt = block_table + (size_t)b * pps;
    const int wd = warp * 64;
    const int k_off = gid * row_h + kvh * FD_HD + wd + 16 * tig;   // token gid, this thread's 16 dims
    const int v_off = kvh * FD_HD + wd + 2 * gid;                  // + 16 mt, token rows 8 ks + 2 tig + e

    uint4 kr[2][2];            // [token gid / gid + 8][dims +0 / +8]
    uint32_t vr[2][2][4];      // [ks][e][mt]
    int cur_page = -1;
    size_t page_row = 0;
    auto load_chunk = [&](int c, uint4 (&kr_)[2][2], uint32_t (&vr_)[2][2][4])
    {
        const int pg = c / FD_PAGE;
        if (pg != cur_page) { cur_page = pg; page_row = (size_t)__ldg(bt + pg) * FD_PAGE; }
        const size_t row0c = page_row + (c % FD_PAGE);
        const half* kb = kc + row0c * row_h;
        const half* vb = vc + row0c * row_h;
        const bool full = c + FD_CHUNK <= t1;
        const uint4 z4 = make_uint4(0, 0, 0, 0);
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            const bool ok = full || c + gid + 8 * h < t1;
            #pragma unroll
            for (int p = 0; p < 2; ++p)
                kr_[h][p] = ok ? __ldg(reinterpret_cast<const uint4*>(kb + k_off + h * 8 * row_h + p * 8)) : z4;
        }
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks)
            #pragma unroll
            for (int e = 0; e < 2; ++e)
            {
                const int t = 8 * ks + 2 * tig + e;
                const bool ok = full || c + t < t1;
                #pragma unroll
                for (int mt = 0; mt < 4; ++mt)
                    vr_[ks][e][mt] = ok ? __ldg(reinterpret_cast<const uint32_t*>(vb + t * row_h + v_off + 16 * mt)) : 0u;
            }
    };
    load_chunk(t0, kr, vr);

    // q rows of this kv head, 8 at a time, scaled by softmax scale * log2(e), natural dim order
    uint32_t qf[NT][8];
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt)
    {
        for (int i = tid; i < 8 * FD_HD; i += FD_THREADS)
        {
            const int rl = i / FD_HD, d = i % FD_HD, r = nt * 8 + rl;
            float v = 0.0f;
            if (r < R)
            {
                const int qi = (row0 + r) / G, hg = (row0 + r) % G;
                v = __half2float(q[(((size_t)b * ql + qi) * nq + kvh * G + hg) * FD_HD + d]) * scale_log2;
            }
            qs[i] = __float2half(v);
        }
        __syncthreads();
        const half* qr = qs + gid * FD_HD + wd + 16 * tig;
        #pragma unroll
        for (int s = 0; s < 8; ++s)
            qf[nt][s] = *reinterpret_cast<const uint32_t*>(qr + 8 * (s >> 2) + 2 * (s & 3));
        __syncthreads();
    }

    float oacc[4][NT][4];
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) oacc[mt][nt][i] = 0.0f;

    const int sm_row_raw = tid / TPR, sm_sub = tid % TPR;
    const bool sm_valid = tid < RP * TPR;
    const bool sm_active = warp * 32 < RP * TPR;
    const int sm_row = sm_valid ? sm_row_raw : RP - 1;
    const int sm_qi = (row0 + sm_row) / G;
    float m_run = -INFINITY, l_run = 0.0f;

    for (int c = t0; c < t1; c += FD_CHUNK)
    {
        uint4 nkr[2][2];
        uint32_t nvr[2][2][4];
        const bool has_next = c + FD_CHUNK < t1;
        if (has_next) load_chunk(c + FD_CHUNK, nkr, nvr);

        float sacc[NT][4];
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) sacc[nt][i] = 0.0f;
        #pragma unroll
        for (int s = 0; s < 8; ++s)
        {
            const uint32_t a0 = reinterpret_cast<const uint32_t*>(&kr[0][s >> 2])[s & 3];
            const uint32_t a1 = reinterpret_cast<const uint32_t*>(&kr[1][s >> 2])[s & 3];
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) mma_1688(sacc[nt], a0, a1, qf[nt][s]);
        }
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
        {
            int col = nt * 8 + 2 * tig;
            sp[warp][gid][col] = sacc[nt][0];
            sp[warp][gid][col + 1] = sacc[nt][1];
            sp[warp][gid + 8][col] = sacc[nt][2];
            sp[warp][gid + 8][col + 1] = sacc[nt][3];
        }
        __syncthreads();

        if (sm_active)
        {
            float sv_[TOK_PER];
            float mc = -INFINITY;
            const int limit = noncausal ? t1 : min(t1, seqlen + sm_qi + 1);
            const int lower = win_left >= 0 ? seqlen + sm_qi - win_left : 0;
            #pragma unroll
            for (int i = 0; i < TOK_PER; ++i)
            {
                int k = sm_sub + i * TPR;
                float s = 0.0f;
                #pragma unroll
                for (int w = 0; w < FD_WARPS; ++w) s += sp[w][k][sm_row];
                s = (sm_valid && sm_row < R && c + k < limit && c + k >= lower) ? s : -INFINITY;
                sv_[i] = s;
                mc = fmaxf(mc, s);
            }
            #pragma unroll
            for (int off = TPR / 2; off > 0; off >>= 1) mc = fmaxf(mc, __shfl_xor_sync(0xffffffffu, mc, off));
            float m_new = fmaxf(m_run, mc);
            float al = (m_new == -INFINITY) ? 1.0f : exp2f(m_run - m_new);
            float lsum = 0.0f;
            #pragma unroll
            for (int i = 0; i < TOK_PER; ++i)
            {
                float pv = (m_new == -INFINITY) ? 0.0f : exp2f(sv_[i] - m_new);
                lsum += pv;
                if (sm_valid) ps[sm_row][sm_sub + i * TPR] = __float2half(pv);
            }
            #pragma unroll
            for (int off = TPR / 2; off > 0; off >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
            l_run = l_run * al + lsum;
            m_run = m_new;
            if (sm_valid && sm_sub == 0) alpha_s[sm_row] = al;
        }
        __syncthreads();

        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
        {
            float a0 = alpha_s[nt * 8 + 2 * tig], a1 = alpha_s[nt * 8 + 2 * tig + 1];
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt)
            {
                oacc[mt][nt][0] *= a0; oacc[mt][nt][1] *= a1;
                oacc[mt][nt][2] *= a0; oacc[mt][nt][3] *= a1;
            }
        }
        uint32_t pb[2][NT];
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
            {
                pb[ks][nt] = *reinterpret_cast<const uint32_t*>(&ps[nt * 8 + gid][ks * 8 + 2 * tig]);
            }
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt)
        {
            #pragma unroll
            for (int ks = 0; ks < 2; ++ks)
            {
                // tokens (2 tig, 2 tig + 1): low halves = dim-row gid, high halves = dim-row gid + 8
                const uint32_t a0 = __byte_perm(vr[ks][0][mt], vr[ks][1][mt], 0x5410);
                const uint32_t a1 = __byte_perm(vr[ks][0][mt], vr[ks][1][mt], 0x7632);
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) mma_1688(oacc[mt][nt], a0, a1, pb[ks][nt]);
            }
        }

        if (has_next)
        {
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int p = 0; p < 2; ++p) kr[h][p] = nkr[h][p];
            #pragma unroll
            for (int ks = 0; ks < 2; ++ks)
                #pragma unroll
                for (int e = 0; e < 2; ++e)
                    #pragma unroll
                    for (int mt = 0; mt < 4; ++mt) vr[ks][e][mt] = nvr[ks][e][mt];
        }
    }

    // partials: accumulator rows gid / gid + 8 -> dims d / d + 1, d = wd + 16 mt + 2 gid
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt)
    {
        int r0 = nt * 8 + 2 * tig;
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt)
        {
            int d0 = wd + 16 * mt + 2 * gid;
            if (r0 < R)
                *reinterpret_cast<float2*>(&part_o[(pbase + r0) * FD_HD + d0]) = make_float2(oacc[mt][nt][0], oacc[mt][nt][2]);
            if (r0 + 1 < R)
                *reinterpret_cast<float2*>(&part_o[(pbase + r0 + 1) * FD_HD + d0]) = make_float2(oacc[mt][nt][1], oacc[mt][nt][3]);
        }
    }
    if (sm_valid && sm_sub == 0 && sm_row < R)
    {
        part_ml[(pbase + sm_row) * 2 + 0] = m_run;
        part_ml[(pbase + sm_row) * 2 + 1] = l_run;
    }
}

template <bool ROT = true>   // false: fd16 partials, which live in the unrotated domain
static __device__ __forceinline__ void fdq4_combine_body(const int r, const int kvh, const int b,
    const float* __restrict__ part_o, const float* __restrict__ part_ml, half* __restrict__ out,
    int splits, int ql, int nq, int nkv)
{
    const int d = threadIdx.x;
    const int G = nq / nkv, R = ql * G;
    const size_t base = ((size_t)b * nkv + kvh) * splits;
    float M = -INFINITY;
    for (int s = 0; s < splits; ++s) M = fmaxf(M, part_ml[((base + s) * R + r) * 2]);
    float L = 0.0f, o = 0.0f;
    for (int s = 0; s < splits; ++s)
    {
        float m = part_ml[((base + s) * R + r) * 2];
        if (m == -INFINITY) continue;
        float w = exp2f(m - M);
        L += w * part_ml[((base + s) * R + r) * 2 + 1];
        o += w * part_o[((base + s) * R + r) * FD_HD + d];
    }
    int qi = r / G, hg = r % G;
    if constexpr (!ROT)
    {
        out[(((size_t)b * ql + qi) * nq + kvh * G + hg) * FD_HD + d] = __float2half((L > 0.0f) ? o / L : 0.0f);
        return;
    }
    __shared__ float os[FD_HD];
    os[d] = (L > 0.0f) ? o / L : 0.0f;
    __syncthreads();
    int g0 = d & ~31, j0 = d & 31;
    float y = 0.0f;
    #pragma unroll 8
    for (int j = 0; j < 32; ++j) y += os[g0 + j] * hsign(j0, j);
    y *= 0.17677669529663687f;
    out[(((size_t)b * ql + qi) * nq + kvh * G + hg) * FD_HD + d] = __float2half(y);
}

