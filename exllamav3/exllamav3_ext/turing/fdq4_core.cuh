#pragma once

// fdq4: flash-decoding attention straight from ExLlamaV3's packed 4-bit K/V cache, tuned for
// Turing (sm_75) and GQA models with head_dim 256 (Qwen3.5/3.8: 24 q heads, 4 kv heads).
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

#define FD_WARPS 4
#define FD_THREADS (FD_WARPS * 32)
#define FD_CHUNK 16
#define FD_HD 256
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
    const int row0, const int nrows)       // this block's q rows [row0, row0 + nrows) of the R = ql * G
{
    constexpr int RP = NT * 8;
    constexpr int TPR = (RP <= 8) ? 16 : (RP <= 16) ? 8 : (RP <= 32) ? 4 : 2;   // softmax threads per row
    constexpr int TOK_PER = FD_CHUNK / TPR;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gid = lane >> 2, tig = lane & 3;
    const int G = nq / nkv, Rt = ql * G, R = nrows;   // Rt: rows in the partial layout, R: rows of this block
    const int row_words = nkv * FD_HD / 8, row_groups = nkv * FD_HD / 32;
    const int seqlen = cache_seqlens[b];
    const int kv_len = seqlen + pre_appended;
    const int t0 = split * split_len;
    const int t1 = min(t0 + split_len, kv_len);

    // One static buffer, two phases: during setup it holds 8 q rows at a time (rotated, scaled,
    // fragment-ordered) while each warp lifts them into registers; afterwards the same bytes hold the
    // per-chunk partial scores, probabilities and rescale factors. Keeps NT = 6 at ~14 KB of smem.
    constexpr int SP_BYTES = FD_WARPS * FD_CHUNK * (RP + 1) * 4;
    constexpr int PS_BYTES = FD_CHUNK * RP * 2;
    constexpr int AL_BYTES = RP * 4;
    constexpr int LOOP_BYTES = SP_BYTES + PS_BYTES + AL_BYTES;
    constexpr int QS_BYTES = 8 * FD_HD * 2;
    __shared__ __align__(16) unsigned char smem_raw[LOOP_BYTES > QS_BYTES ? LOOP_BYTES : QS_BYTES];
    half* qs = reinterpret_cast<half*>(smem_raw);
    auto sp = reinterpret_cast<float (*)[FD_CHUNK][RP + 1]>(smem_raw);
    auto ps = reinterpret_cast<half (*)[RP]>(smem_raw + SP_BYTES);
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
            const int limit = min(t1, seqlen + sm_qi + 1);   // causal
            #pragma unroll
            for (int i = 0; i < TOK_PER; ++i)
            {
                int k = sm_sub + i * TPR;
                float s = sp[0][k][sm_row] + sp[1][k][sm_row] + sp[2][k][sm_row] + sp[3][k][sm_row];
                s = (sm_valid && sm_row < R && c + k < limit) ? s : -INFINITY;
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
                if (sm_valid) ps[sm_sub + i * TPR][sm_row] = __float2half(pv);
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
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks)
        {
            uint32_t pb[NT];
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
            {
                __half2 p2 = __halves2half2(ps[ks * 8 + 2 * tig][nt * 8 + gid], ps[ks * 8 + 2 * tig + 1][nt * 8 + gid]);
                pb[nt] = *reinterpret_cast<uint32_t*>(&p2);
            }
            __half2 sc = __halves2half2(__hmul(vsc[ks][0], eighth), __hmul(vsc[ks][1], eighth));
            __half2 bi = __hmul2(sc, m05);
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt)
            {
                // byte mt of both tokens' words -> 16-bit lanes (tok0, tok1); lo nibble = dim-row gid, hi = gid+8
                uint32_t t = __byte_perm(vwd[ks][0], vwd[ks][1], 0x4400 + mt * 0x1111);
                uint32_t a0 = deq2(lop3_and_or(t, 0x000F000Fu, 0x64006400u), sc, bi);
                uint32_t a1 = deq2(lop3_and_or(t >> 4, 0x000F000Fu, 0x64006400u), sc, bi);
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) mma_1688(oacc[mt][nt], a0, a1, pb[nt]);
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
    __shared__ float os[FD_HD];
    os[d] = (L > 0.0f) ? o / L : 0.0f;
    __syncthreads();
    int g0 = d & ~31, j0 = d & 31;
    float y = 0.0f;
    #pragma unroll 8
    for (int j = 0; j < 32; ++j) y += os[g0 + j] * hsign(j0, j);
    y *= 0.17677669529663687f;
    int qi = r / G, hg = r % G;
    out[(((size_t)b * ql + qi) * nq + kvh * G + hg) * FD_HD + d] = __float2half(y);
}

