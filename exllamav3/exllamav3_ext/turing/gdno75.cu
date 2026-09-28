// gdno75: gated delta rule output stage (FLA's chunk_fwd_kernel_o) on Turing HMMA. Triton runs it as scalar FMA with
// heavy spills on sm_75 (22 ms per 2K-token call at Qwen3.8-27B shapes); the batched-cuBLAS form takes 2.6 ms, most
// of it in the copies that expand q and k to the value heads.
//
// Per 64-token chunk and value head:
//   o = scale * [ (q @ h) * exp2(g_i) + tril_incl((q @ k^T) * exp2(g_i - g_j)) @ v ]
// One block per (chunk, key head), 4 warps x 16 rows, running the key head's value heads in turn: q stays in mma A
// fragments and q k^T in accumulators across them; per value head the 128x128 state h (32 KB) and then the V tile
// stream through one shared-memory region (swizzled, ldmatrix), and P comes straight from the accumulators (the
// m16n8 accumulator layout is the m16n8k8 A layout). 32 KB per block: two blocks per SM.
//
// Scope: K = V = 128, chunk 64, fp16 q/k [B,T,H,K], v [B,T,HV,V], h [B,NT,HV,K,V]; fp32 g [B,T,HV] (chunk-local
// cumsum, log2 domain); no cu_seqlens, state layout [K, V].
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
#define ROW_CHUNKS 16                   // 16 x 16-byte chunks per 256-byte fp16 row (K = V = 128)
#define SMEM_BYTES 32768

static __device__ __forceinline__ uint32_t tile_off(int r, int c) { return (uint32_t)((r * ROW_CHUNKS + (c ^ (r & 7))) * 16); }

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

// rows x 128 fp16 rows (row_stride elements apart) -> shared tile at byte offset `off`; rows past `valid` read as 0
static __device__ __forceinline__ void load_rows(uint8_t* sm, uint32_t off, const half* src, int64_t row_stride, int rows, int valid, int t_id)
{
    for (int idx = t_id; idx < rows * ROW_CHUNKS; idx += NTHREADS)
    {
        int r = idx / ROW_CHUNKS, c = idx % ROW_CHUNKS;
        uint4 val = r < valid ? *reinterpret_cast<const uint4*>(src + (int64_t) r * row_stride + c * 8) : make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(sm + off + tile_off(r, c)) = val;
    }
}

__global__ void __launch_bounds__(NTHREADS, 2)
gdno75_kernel
(
    const half* __restrict__ q,      // [B, T, H, K]
    const half* __restrict__ k,      // [B, T, H, K]
    const half* __restrict__ v,      // [B, T, HV, V]
    const half* __restrict__ h,      // [B, NT, HV, K, V]
    const float* __restrict__ g,     // [B, T, HV]
    half* __restrict__ o,            // [B, T, HV, V]
    float scale,
    int T, int H, int HV
)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
    __trap();
#else
    __shared__ __align__(128) uint8_t sm[SMEM_BYTES];
    const uint32_t sbase = (uint32_t) __cvta_generic_to_shared(sm);

    const int t_id = threadIdx.x;
    const int warp = t_id >> 5;
    const int lane = t_id & 31;
    const int gq = lane >> 2;
    const int cq = lane & 3;
    const int lr = lane & 7;
    const int lm = lane >> 3;

    const int it = blockIdx.x;
    const int i_nk = blockIdx.y;
    const int i_n = i_nk / H;
    const int i_hk = i_nk % H;
    const int G = HV / H;
    const int NT = (T + BT - 1) / BT;
    const int t0 = it * BT;
    const int rows = min(BT, T - t0);
    const int64_t bos = (int64_t) i_n * T;

    // q -> [0, 16K), k -> [16K, 32K)
    load_rows(sm, 0, q + ((bos + t0) * H + i_hk) * KD, (int64_t) H * KD, BT, rows, t_id);
    load_rows(sm, 16384, k + ((bos + t0) * H + i_hk) * KD, (int64_t) H * KD, BT, rows, t_id);
    __syncthreads();

    // q A fragments (rows 16w..16w+15) and S = q k^T (8 n8 tiles over the 64 keys)
    uint32_t qf[KD / 8][2];
    float s[8][4];
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt)
        #pragma unroll
        for (int i = 0; i < 4; ++i) s[nt][i] = 0.0f;
    #pragma unroll
    for (int kk = 0; kk < KD / 8; kk += 2)
    {
        uint32_t a[4];
        ldsm_x4(a, sbase + tile_off(16 * warp + lr + (lm & 1) * 8, kk + (lm >> 1)));
        qf[kk][0] = a[0]; qf[kk][1] = a[1]; qf[kk + 1][0] = a[2]; qf[kk + 1][1] = a[3];
        #pragma unroll
        for (int np = 0; np < 4; ++np)
        {
            uint32_t b[4];
            ldsm_x4(b, sbase + 16384 + tile_off(16 * np + 8 * (lm & 1) + lr, kk + (lm >> 1)));
            mma1688(s[2 * np], a[0], a[1], b[0]);
            mma1688(s[2 * np + 1], a[0], a[1], b[1]);
            mma1688(s[2 * np], a[2], a[3], b[2]);
            mma1688(s[2 * np + 1], a[2], a[3], b[3]);
        }
    }

    const int r0 = 16 * warp + gq;
    for (int gi = 0; gi < G; ++gi)
    {
        const int i_h = i_hk * G + gi;
        const float* g_ = g + (bos + t0) * HV + i_h;
        auto gate = [&] (int t) -> float { return t < rows ? g_[(int64_t) t * HV] : 0.0f; };
        const float g_r0 = gate(r0), g_r1 = gate(r0 + 8);

        __syncthreads();                                        // previous tiles no longer read
        load_rows(sm, 0, h + (((int64_t) i_n * NT + it) * HV + i_h) * KD * VD, VD, KD, KD, t_id);
        __syncthreads();

        // acc = q @ h (B = h[k][v]: rows = k, via ldmatrix.trans)
        float acc[VD / 8][4];
        #pragma unroll
        for (int nt = 0; nt < VD / 8; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[nt][i] = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < KD / 8; ++kk)
        {
            #pragma unroll
            for (int dn = 0; dn < VD / 8; dn += 4)
            {
                uint32_t b[4];
                ldsm_x4_t(b, sbase + tile_off(8 * kk + lr, dn + lm));
                #pragma unroll
                for (int x = 0; x < 4; ++x) mma1688(acc[dn + x], qf[kk][0], qf[kk][1], b[x]);
            }
        }
        const float e0 = exp2f(g_r0), e1 = exp2f(g_r1);
        #pragma unroll
        for (int nt = 0; nt < VD / 8; ++nt)
        {
            acc[nt][0] *= e0; acc[nt][1] *= e0;
            acc[nt][2] *= e1; acc[nt][3] *= e1;
        }

        __syncthreads();                                        // h no longer read
        load_rows(sm, 0, v + ((bos + t0) * HV + i_h) * VD, (int64_t) HV * VD, BT, rows, t_id);
        __syncthreads();

        // acc += tril_incl(S * exp2(g_i - g_j)) @ v, P fragments straight from S
        #pragma unroll
        for (int j8 = 0; j8 < BT / 8; ++j8)
        {
            float p[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                int i = r0 + (e >> 1) * 8;
                int j = 8 * j8 + 2 * cq + (e & 1);
                p[e] = j <= i ? s[j8][e] * exp2f((e >> 1 ? g_r1 : g_r0) - gate(j)) : 0.0f;
            }
            uint32_t a0 = pack_h2(p[0], p[1]);
            uint32_t a1 = pack_h2(p[2], p[3]);
            #pragma unroll
            for (int dn = 0; dn < VD / 8; dn += 4)
            {
                uint32_t b[4];
                ldsm_x4_t(b, sbase + tile_off(8 * j8 + lr, dn + lm));
                #pragma unroll
                for (int x = 0; x < 4; ++x) mma1688(acc[dn + x], a0, a1, b[x]);
            }
        }

        half* o_ = o + ((bos + t0) * HV + i_h) * VD;
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh)
        {
            int r = r0 + hh * 8;
            if (r < rows)
            {
                #pragma unroll
                for (int nt = 0; nt < VD / 8; ++nt)
                    *reinterpret_cast<uint32_t*>(o_ + (int64_t) r * HV * VD + 8 * nt + 2 * cq) =
                        pack_h2(acc[nt][2 * hh] * scale, acc[nt][2 * hh + 1] * scale);
            }
        }
    }
#endif
}

void gdno75_fwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor h, at::Tensor g, at::Tensor o, double scale)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    TORCH_CHECK(q.dtype() == at::kHalf && k.dtype() == at::kHalf && v.dtype() == at::kHalf && h.dtype() == at::kHalf &&
                o.dtype() == at::kHalf, "fp16 q/k/v/h/o");
    TORCH_CHECK(g.dtype() == at::kFloat, "fp32 g");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && h.is_contiguous() && g.is_contiguous() &&
                o.is_contiguous(), "contiguous");
    int B = q.size(0), T = q.size(1), H = q.size(2), HV = v.size(2);
    TORCH_CHECK(q.size(3) == KD && k.size(3) == KD && v.size(3) == VD && HV % H == 0, "K = V = 128");
    int NT = (T + BT - 1) / BT;
    TORCH_CHECK(h.size(1) == NT && h.size(2) == HV && h.size(3) == KD && h.size(4) == VD, "h shape");
    dim3 grid(NT, B * H);
    gdno75_kernel<<<grid, NTHREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const half*) q.data_ptr(), (const half*) k.data_ptr(), (const half*) v.data_ptr(), (const half*) h.data_ptr(),
        g.data_ptr<float>(), (half*) o.data_ptr(), (float) scale, T, H, HV);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#else

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

void gdno75_fwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor h, at::Tensor g, at::Tensor o, double scale)
{
    TORCH_CHECK(false, "Turing (sm_75) kernel not available on ROCm");
}

#endif
