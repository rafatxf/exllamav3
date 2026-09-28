// gdnwy75: gated delta rule WY representation (FLA's fused kkt + solve_tril + recompute_w_u) on Turing HMMA.
// Triton runs these tl.dot products as scalar FMA on sm_75 (~2 ms per 2K-token call at Qwen3.8-27B shapes).
//
// Per 64-token chunk and value head (one block, 4 warps):
//   A   = strict_tril(K K^T * exp2(g_i - g_j)) * beta_i                  K K^T on mma.m16n8k8, fp32
//   X   = (I + A)^-1                                                     unit lower triangular, fp32
//   u   = (X diag(beta)) @ V                                            mma, fp16 in / fp32 acc / fp16 out
//   w   = (X diag(beta * exp2(g))) @ K
// X is built blockwise as in FLA's solve_tril: the four 16x16 diagonal blocks by forward substitution in
// registers, then the 32x32 and 64x64 merges X_hl = -X_hh A_hl X_ll as fp32 products spread over the block. The
// column scales ride on X's A fragments instead of rescaling the K / V tiles. Shared memory is two 16 KB regions
// (K tile, then A, then V, then K again; and X), so two blocks fit per SM.
//
// Scope: K = V = 128, chunk 64, fp16 k [B,T,H,K], v [B,T,HV,V], beta [B,T,HV]; fp32 g [B,T,HV] (chunk-local
// cumsum, log2 domain); no cu_seqlens.
#if !defined(USE_ROCM)

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#define KD 128
#define VD 128
#define BT 64
#define NWARPS 4
#define NTHREADS (NWARPS * 32)
#define ROW_CHUNKS (KD / 8)             // 16 x 16-byte chunks per 256-byte fp16 row

#define R0_OFF 0                        // K tile / A (fp32 64x64) / V tile / K tile
#define X_OFF 16384                     // X (fp32 64x64)
#define SMEM_BYTES 32768

static __device__ __forceinline__ uint32_t tile_off(int r, int c) { return (uint32_t)((r * ROW_CHUNKS + (c ^ (r & 7))) * 16); }
// fp32 64x64 matrices: word swizzle keeps the 8 rows of an mma fragment on distinct banks and pairs (c, c+1) adjacent
static __device__ __forceinline__ int sidx(int r, int c) { return r * 64 + (c ^ ((r & 7) << 2)); }

static __device__ __forceinline__ void mma1688(float* c, uint32_t a0, uint32_t a1, uint32_t b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b));
}

static __device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t saddr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(saddr));
}

static __device__ __forceinline__ void ldsm_x4_t(uint32_t* r, uint32_t saddr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(saddr));
}

static __device__ __forceinline__ uint32_t pack_h2(float a, float b)
{
    half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}

static __device__ __forceinline__ void load_tile(uint8_t* sm, const half* src, int64_t row_stride, int rows, int t_id)
{
    #pragma unroll
    for (int j = 0; j < (BT * ROW_CHUNKS) / NTHREADS; ++j)
    {
        int idx = t_id + j * NTHREADS;
        int r = idx / ROW_CHUNKS, c = idx % ROW_CHUNKS;
        uint4 val = r < rows ? *reinterpret_cast<const uint4*>(src + (int64_t) r * row_stride + c * 8) : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(sm + R0_OFF + tile_off(r, c)) = val;
    }
}

__global__ void __launch_bounds__(NTHREADS, 2)
gdnwy75_kernel
(
    const half* __restrict__ k,      // [B, T, H, K]
    const half* __restrict__ v,      // [B, T, HV, V]
    const half* __restrict__ beta,   // [B, T, HV]
    const float* __restrict__ g,     // [B, T, HV]
    half* __restrict__ w,            // [B, T, HV, K]
    half* __restrict__ u,            // [B, T, HV, V]
    int T, int H, int HV
)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
    __trap();
#else
    __shared__ __align__(128) uint8_t sm[SMEM_BYTES];
    const uint32_t sbase = (uint32_t) __cvta_generic_to_shared(sm);
    float* As = reinterpret_cast<float*>(sm + R0_OFF);
    float* Xs = reinterpret_cast<float*>(sm + X_OFF);

    const int t_id = threadIdx.x;
    const int warp = t_id >> 5;
    const int lane = t_id & 31;
    const int gq = lane >> 2;
    const int cq = lane & 3;
    const int lr = lane & 7;
    const int lm = lane >> 3;

    const int it = blockIdx.x;
    const int i_nh = blockIdx.y;
    const int i_n = i_nh / HV;
    const int i_h = i_nh % HV;
    const int i_hk = i_h / (HV / H);
    const int t0 = it * BT;
    const int rows = min(BT, T - t0);
    const int64_t bos = (int64_t) i_n * T;

    const half* k_ = k + ((bos + t0) * H + i_hk) * KD;          // + t * H * KD
    const half* v_ = v + ((bos + t0) * HV + i_h) * VD;          // + t * HV * VD
    const half* b_ = beta + (bos + t0) * HV + i_h;              // + t * HV
    const float* g_ = g + (bos + t0) * HV + i_h;
    half* w_ = w + ((bos + t0) * HV + i_h) * KD;
    half* u_ = u + ((bos + t0) * HV + i_h) * VD;
    auto gate = [&] (int t) -> float { return t < rows ? g_[(int64_t) t * HV] : 0.0f; };
    auto bet = [&] (int t) -> float { return t < rows ? __half2float(b_[(int64_t) t * HV]) : 0.0f; };

    load_tile(sm, k_, (int64_t) H * KD, rows, t_id);
    for (int i = t_id; i < BT * BT; i += NTHREADS) Xs[i] = 0.0f;
    __syncthreads();

    // S = K K^T, warp rows 16w..16w+15, 8 n8 tiles over the 64 keys
    float s[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt)
        #pragma unroll
        for (int i = 0; i < 4; ++i) s[nt][i] = 0.0f;
    #pragma unroll
    for (int kk = 0; kk < KD / 8; kk += 2)
    {
        uint32_t a[4];
        ldsm_x4(a, sbase + R0_OFF + tile_off(16 * warp + lr + (lm & 1) * 8, kk + (lm >> 1)));
        #pragma unroll
        for (int np = 0; np < 4; ++np)
        {
            // matrices: (keys 16np+0..7, chunk kk), (16np+8..15, kk), (16np+0..7, kk+1), (16np+8..15, kk+1)
            uint32_t b[4];
            ldsm_x4(b, sbase + R0_OFF + tile_off(16 * np + 8 * (lm & 1) + lr, kk + (lm >> 1)));
            mma1688(s[2 * np], a[0], a[1], b[0]);
            mma1688(s[2 * np + 1], a[0], a[1], b[1]);
            mma1688(s[2 * np], a[2], a[3], b[2]);
            mma1688(s[2 * np + 1], a[2], a[3], b[3]);
        }
    }
    __syncthreads();                                           // K tile no longer read: region 0 takes A

    // A = strict_tril(S * exp2(g_i - g_j)) * beta_i (the upper triangle is scratch for the merges below)
    {
        float gi[2], bi[2];
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) { gi[hh] = gate(16 * warp + gq + hh * 8); bi[hh] = bet(16 * warp + gq + hh * 8); }
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt)
            #pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                int i = 16 * warp + gq + (e >> 1) * 8;
                int j = 8 * nt + 2 * cq + (e & 1);
                float a = 0.0f;
                if (j < i && i < rows) a = s[nt][e] * exp2f(gi[e >> 1] - gate(j)) * bi[e >> 1];
                As[sidx(i, j)] = a;
            }
    }
    __syncthreads();

    // X diagonal blocks: thread (block bd, column c) solves (I + A_dd) x = e_c in registers
    if (t_id < BT)
    {
        const int bd = t_id >> 4, c = t_id & 15, o = 16 * bd;
        float x[16];
        #pragma unroll
        for (int r = 0; r < 16; ++r) x[r] = r == c ? 1.0f : 0.0f;
        #pragma unroll
        for (int r = 1; r < 16; ++r)
        {
            float acc = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < r; ++kk) acc = fmaf(As[sidx(o + r, o + kk)], x[kk], acc);
            if (r > c) x[r] = -acc;
        }
        #pragma unroll
        for (int r = 0; r < 16; ++r) Xs[sidx(o + r, o + c)] = x[r];
    }
    __syncthreads();

    // 32x32 merges, pairs (0,1) and (2,3): X_hl = -X_hh (A_hl X_ll); temporaries in A's upper triangle
    {
        const int p = t_id >> 6, idx = t_id & 63;
        const int lo = 32 * p, hi = lo + 16;
        const int r = idx >> 2, c0 = (idx & 3) * 4;
        float tmp[4] = {};
        for (int kk = 0; kk < 16; ++kk)
        {
            float a = As[sidx(hi + r, lo + kk)];
            #pragma unroll
            for (int q = 0; q < 4; ++q) tmp[q] = fmaf(a, Xs[sidx(lo + kk, lo + c0 + q)], tmp[q]);
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) As[sidx(lo + r, hi + c0 + q)] = tmp[q];         // block (lo, hi): upper, free
        __syncthreads();
        float out[4] = {};
        for (int kk = 0; kk <= r; ++kk)
        {
            float xh = Xs[sidx(hi + r, hi + kk)];
            #pragma unroll
            for (int q = 0; q < 4; ++q) out[q] = fmaf(xh, As[sidx(lo + kk, hi + c0 + q)], out[q]);
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) Xs[sidx(hi + r, lo + c0 + q)] = -out[q];
    }
    __syncthreads();

    // 64x64 merge: X[32:64][0:32] = -X[32:64][32:64] (A[32:64][0:32] X[0:32][0:32]); temporary in A[0:32][32:64]
    {
        const int r = t_id >> 2, c0 = (t_id & 3) * 8;
        float tmp[8] = {};
        for (int kk = 0; kk < 32; ++kk)
        {
            float a = As[sidx(32 + r, kk)];
            #pragma unroll
            for (int q = 0; q < 8; ++q) tmp[q] = fmaf(a, Xs[sidx(kk, c0 + q)], tmp[q]);
        }
        #pragma unroll
        for (int q = 0; q < 8; ++q) As[sidx(r, 32 + c0 + q)] = tmp[q];
        __syncthreads();
        float out[8] = {};
        for (int kk = 0; kk <= r; ++kk)
        {
            float xh = Xs[sidx(32 + r, 32 + kk)];
            #pragma unroll
            for (int q = 0; q < 8; ++q) out[q] = fmaf(xh, As[sidx(kk, 32 + c0 + q)], out[q]);
        }
        #pragma unroll
        for (int q = 0; q < 8; ++q) Xs[sidx(32 + r, c0 + q)] = -out[q];
    }
    __syncthreads();                                           // X complete; A no longer read

    // out = (X diag(scale)) @ tile, with the tile's rows = keys (the k dim) and columns = features
    const int r0 = 16 * warp + gq;
    #pragma unroll 1
    for (int pass = 0; pass < 2; ++pass)
    {
        if (pass == 0) load_tile(sm, v_, (int64_t) HV * VD, rows, t_id);
        else load_tile(sm, k_, (int64_t) H * KD, rows, t_id);
        __syncthreads();

        float acc[16][4];
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[nt][i] = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < BT / 8; ++kk)
        {
            int c0 = 8 * kk + 2 * cq;
            float s0 = bet(c0), s1 = bet(c0 + 1);
            if (pass == 1) { s0 *= exp2f(gate(c0)); s1 *= exp2f(gate(c0 + 1)); }
            uint32_t a0 = pack_h2(Xs[sidx(r0, c0)] * s0, Xs[sidx(r0, c0 + 1)] * s1);
            uint32_t a1 = pack_h2(Xs[sidx(r0 + 8, c0)] * s0, Xs[sidx(r0 + 8, c0 + 1)] * s1);
            #pragma unroll
            for (int dn = 0; dn < 16; dn += 4)
            {
                uint32_t b[4];
                ldsm_x4_t(b, sbase + R0_OFF + tile_off(8 * kk + lr, dn + lm));
                #pragma unroll
                for (int x = 0; x < 4; ++x) mma1688(acc[dn + x], a0, a1, b[x]);
            }
        }
        half* out = pass == 0 ? u_ : w_;
        const int64_t ostride = (int64_t) HV * (pass == 0 ? VD : KD);
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh)
        {
            int r = r0 + hh * 8;
            if (r < rows)
            {
                #pragma unroll
                for (int nt = 0; nt < 16; ++nt)
                    *reinterpret_cast<uint32_t*>(out + r * ostride + 8 * nt + 2 * cq) = pack_h2(acc[nt][2 * hh], acc[nt][2 * hh + 1]);
            }
        }
        __syncthreads();                                       // tile no longer read
    }
#endif
}

void gdnwy75_fwd(at::Tensor k, at::Tensor v, at::Tensor beta, at::Tensor g, at::Tensor w, at::Tensor u)
{
    const at::cuda::OptionalCUDAGuard guard(k.device());
    TORCH_CHECK(k.dtype() == at::kHalf && v.dtype() == at::kHalf && beta.dtype() == at::kHalf, "fp16 k/v/beta");
    TORCH_CHECK(g.dtype() == at::kFloat, "fp32 g");
    TORCH_CHECK(w.dtype() == at::kHalf && u.dtype() == at::kHalf, "fp16 w/u");
    TORCH_CHECK(k.is_contiguous() && v.is_contiguous() && beta.is_contiguous() && g.is_contiguous() &&
                w.is_contiguous() && u.is_contiguous(), "contiguous");
    int B = k.size(0), T = k.size(1), H = k.size(2), HV = v.size(2);
    TORCH_CHECK(k.size(3) == KD && v.size(3) == VD && w.size(3) == KD && HV % H == 0, "K = V = 128");
    int NT = (T + BT - 1) / BT;
    dim3 grid(NT, B * HV);
    gdnwy75_kernel<<<grid, NTHREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const half*) k.data_ptr(), (const half*) v.data_ptr(), (const half*) beta.data_ptr(), g.data_ptr<float>(),
        (half*) w.data_ptr(), (half*) u.data_ptr(), T, H, HV);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#else

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

void gdnwy75_fwd(at::Tensor k, at::Tensor v, at::Tensor beta, at::Tensor g, at::Tensor w, at::Tensor u)
{
    TORCH_CHECK(false, "Turing (sm_75) kernel not available on ROCm");
}

#endif
