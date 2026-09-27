// gdnh75: Gated DeltaNet chunk-state recurrence (FLA chunk_gated_delta_rule_fwd_kernel_h) on Turing
// tensor cores. Triton lowers tl.dot to scalar FMA on sm_75, so the FLA kernel runs at ~1.2 TFLOPS;
// this one keeps the state h in mma.m16n8k8 fp32 accumulators for the whole sequence.
//
// Scope: K = V = 128, BT = 64, fp16 k/w/u, fp32 g (chunk-local cumsum, log2 domain), scalar gate only
// (no gk), state layout [K, V] (state_v_first = False), equal-length sequences (no cu_seqlens).
//
// Per chunk t (one block = (32-wide V slice, one value head), 4 warps):
//   h[t]   = state (stored fp16)
//   v_new  = u - w @ h           (64 x 32, K = 128)    warp w: rows 16w..16w+15
//   vd     = v_new * exp2(g_last - g_row)  (0 for rows past T)
//   state  = state * exp2(g_last) + k^T @ vd   (128 x 32, K = 64 rows)   warp w: state rows 32w..32w+31
//
// Implementation: all shared-memory operands are read with ldmatrix(.trans) from XOR-swizzled, unpadded tiles; the
// w @ h reduction runs over a permuted K order (mma k index 2c+e <-> column 32c + 2kk + e) so every
// thread's w operand is one contiguous 64-byte run (4 x 16-byte loads instead of 32 x 4-byte); the
// fp16 state tile aliases the k tile, so a block needs 20.25 KB and three blocks fit on an SM (the 192
// blocks of a 48-head layer then run in one wave on 68 SMs).
#if !defined(USE_ROCM)

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#define KD 128
#define VD 128
#define BT 64
#define BV 32
#define NWARPS 4
#define NTHREADS (NWARPS * 32)

// Tile layouts, all in 16-byte chunks:
//   ks [t 0..63][kidx 0..127]   16 chunks/row, chunk ^= t & 7
//   hs [kidx 0..127][v 0..31]    4 chunks/row, chunk ^= ((r >> 5) ^ (r >> 1)) & 3   (aliases ks)
//   vs [t 0..63][v 0..31]        4 chunks/row, chunk ^= (t >> 1) & 3
#define KS_OFF 0
#define HS_OFF 0
#define VS_OFF 16384
#define SMEM_TILES 20480

static __device__ __forceinline__ int ks_addr(int t, int c) { return KS_OFF + (t * 16 + (c ^ (t & 7))) * 16; }
static __device__ __forceinline__ int hs_addr(int r, int c) { return HS_OFF + (r * 4 + (c ^ (((r >> 5) ^ (r >> 1)) & 3))) * 16; }
static __device__ __forceinline__ int vs_addr(int t, int c) { return VS_OFF + (t * 4 + (c ^ ((t >> 1) & 3))) * 16; }

static __device__ __forceinline__ void mma1688(float* c, uint32_t a0, uint32_t a1, uint32_t b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b));
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

__global__ void __launch_bounds__(NTHREADS, 3)
gdnh75_kernel
(
    const half* __restrict__ k,      // [B, T, H, K]
    const half* __restrict__ w,      // [B, T, HV, K]
    const half* __restrict__ u,      // [B, T, HV, V]
    const float* __restrict__ g,     // [B, T, HV]
    const float* __restrict__ h0,    // [B, HV, K, V] or null
    half* __restrict__ h,            // [B, NT, HV, K, V]
    half* __restrict__ v_new,        // [B, T, HV, V] or null
    float* __restrict__ ht,          // [B, HV, K, V] or null
    int T, int H, int HV
)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
    __trap();
#else
    __shared__ __align__(128) uint8_t sm[SMEM_TILES];
    __shared__ float gs[BT];
    const uint32_t sbase = (uint32_t) __cvta_generic_to_shared(sm);

    const int t_id = threadIdx.x;
    const int warp = t_id >> 5;
    const int lane = t_id & 31;
    const int gq = lane >> 2;           // mma groupID
    const int cq = lane & 3;            // mma threadID_in_group
    const int lr = lane & 7;            // ldmatrix: row supplied by this lane
    const int lm = lane >> 3;           // ldmatrix: matrix this lane addresses
    const int v0 = blockIdx.x * BV;
    const int i_nh = blockIdx.y;
    const int i_n = i_nh / HV;
    const int i_h = i_nh % HV;
    const int i_hk = i_h / (HV / H);
    const int NT = (T + BT - 1) / BT;
    const int64_t bos = (int64_t) i_n * T;

    const half* k_ = k + (bos * H + i_hk) * KD;             // + t * H * KD + kidx
    const half* w_ = w + (bos * HV + i_h) * KD;             // + t * HV * KD + kidx
    const half* u_ = u + (bos * HV + i_h) * VD + v0;        // + t * HV * VD + v
    const float* g_ = g + bos * HV + i_h;                   // + t * HV
    half* vn_ = v_new ? v_new + (bos * HV + i_h) * VD + v0 : nullptr;
    half* h_ = h + ((int64_t) i_n * NT * HV + i_h) * KD * VD + v0;   // + t * HV * K * V + kidx * V + v

    // State: warp owns rows 32*warp .. +31 as 2 m16 tiles x 4 n8 tiles
    float st[2][4][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                st[mt][nt][i] = 0.0f;
    if (h0)
    {
        const float* h0_ = h0 + (int64_t) i_nh * KD * VD + v0;
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt)
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt)
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh)
                {
                    int r = 32 * warp + 16 * mt + gq + hh * 8;
                    float2 x = *reinterpret_cast<const float2*>(h0_ + r * VD + 8 * nt + 2 * cq);
                    st[mt][nt][2 * hh] = x.x;
                    st[mt][nt][2 * hh + 1] = x.y;
                }
    }

    const int r0 = 16 * warp + gq, r1 = r0 + 8;     // v_new rows of this thread

    for (int it = 0; it < NT; ++it)
    {
        const int t0 = it * BT;
        const int rows = min(BT, T - t0);

        // Issue this chunk's global loads: k tile (8 x 16 B / thread), w run (2 rows x 64 B), u, g
        uint4 kreg[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int idx = t_id + j * NTHREADS;
            int t = idx >> 4, c = idx & 15;
            kreg[j] = t < rows ? *reinterpret_cast<const uint4*>(k_ + (int64_t)(t0 + t) * H * KD + c * 8) : make_uint4(0, 0, 0, 0);
        }
        uint4 wreg[2][4];
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh)
        {
            int r = hh ? r1 : r0;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                wreg[hh][j] = r < rows ? *reinterpret_cast<const uint4*>(w_ + (int64_t)(t0 + r) * HV * KD + 32 * cq + 8 * j) : make_uint4(0, 0, 0, 0);
        }
        half2 ureg[2][4];
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh)
        {
            int r = hh ? r1 : r0;
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt)
                ureg[hh][nt] = r < rows ? *reinterpret_cast<const half2*>(u_ + (int64_t)(t0 + r) * HV * VD + 8 * nt + 2 * cq) : __floats2half2_rn(0.0f, 0.0f);
        }
        if (t_id < BT) gs[t_id] = t_id < rows ? g_[(int64_t)(t0 + t_id) * HV] : 0.0f;

        // State (fp16) -> hs
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt)
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt)
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh)
                {
                    int r = 32 * warp + 16 * mt + gq + hh * 8;
                    *reinterpret_cast<uint32_t*>(sm + hs_addr(r, nt) + cq * 4) = pack_h2(st[mt][nt][2 * hh], st[mt][nt][2 * hh + 1]);
                }
        __syncthreads();

        // h[t] -> global, 16-byte chunks
        {
            half* hto = h_ + (int64_t) it * HV * KD * VD;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                int idx = t_id + j * NTHREADS;
                int r = idx >> 2, c = idx & 3;
                *reinterpret_cast<uint4*>(hto + r * VD + c * 8) = *reinterpret_cast<const uint4*>(sm + hs_addr(r, c));
            }
        }

        // v_new = u - w @ h over the permuted K order: step kk, mma k index j = 2c + e <-> kidx 32c + 2kk + e
        float acc[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) acc[nt][i] = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < KD / 8; ++kk)
        {
            // B fragments for the 4 n8 tiles: lane supplies row kidx(lr) of matrix (n tile) lm
            uint32_t b[4];
            int kr = 32 * (lr >> 1) + 2 * kk + (lr & 1);
            ldsm_x4_t(b, sbase + hs_addr(kr, lm));
            uint32_t a0 = reinterpret_cast<const uint32_t*>(&wreg[0][kk >> 2])[kk & 3];
            uint32_t a1 = reinterpret_cast<const uint32_t*>(&wreg[1][kk >> 2])[kk & 3];
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) mma1688(acc[nt], a0, a1, b[nt]);
        }

        // v_new out, decayed vd -> vs
        const float gl = gs[rows - 1];
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh)
        {
            int r = hh ? r1 : r0;
            bool ok = r < rows;
            float dec = ok ? exp2f(gl - gs[r]) : 0.0f;
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt)
            {
                float2 uu = __half22float2(ureg[hh][nt]);
                float x0 = uu.x - acc[nt][2 * hh], x1 = uu.y - acc[nt][2 * hh + 1];
                if (ok && vn_) *reinterpret_cast<half2*>(vn_ + (int64_t)(t0 + r) * HV * VD + 8 * nt + 2 * cq) = __floats2half2_rn(x0, x1);
                *reinterpret_cast<uint32_t*>(sm + vs_addr(r, nt) + cq * 4) = pack_h2(x0 * dec, x1 * dec);
            }
        }
        __syncthreads();

        // k tile -> ks (overwrites hs)
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            int idx = t_id + j * NTHREADS;
            int t = idx >> 4, c = idx & 15;
            *reinterpret_cast<uint4*>(sm + ks_addr(t, c)) = kreg[j];
        }
        __syncthreads();

        // state = state * exp2(g_last) + k^T @ vd
        {
            const float sc = exp2f(gl);
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt)
                #pragma unroll
                for (int nt = 0; nt < 4; ++nt)
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) st[mt][nt][i] *= sc;
            #pragma unroll
            for (int kk2 = 0; kk2 < BT / 16; ++kk2)          // two k8 steps per iteration
            {
                #pragma unroll
                for (int ks = 0; ks < 2; ++ks)
                {
                    int tb = kk2 * 16 + ks * 8;
                    uint32_t b[4];
                    ldsm_x4_t(b, sbase + vs_addr(tb + lr, lm));
                    #pragma unroll
                    for (int mt = 0; mt < 2; ++mt)
                    {
                        // A = k^T (m = kidx, k = t): matrices (m 0-7, t), (m 8-15, t) -> a0, a1
                        int mb = 32 * warp + 16 * mt;
                        uint32_t a[4];
                        // lanes 0-7: rows t = tb + lr, chunk mb/8; lanes 8-15: chunk mb/8 + 1; 16-31 unused (repeat)
                        ldsm_x4_t(a, sbase + ks_addr(tb + lr, (mb >> 3) + (lm & 1)));
                        #pragma unroll
                        for (int nt = 0; nt < 4; ++nt) mma1688(st[mt][nt], a[0], a[1], b[nt]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (ht)
    {
        float* hto = ht + (int64_t) i_nh * KD * VD + v0;
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt)
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt)
                #pragma unroll
                for (int hh = 0; hh < 2; ++hh)
                {
                    int r = 32 * warp + 16 * mt + gq + hh * 8;
                    *reinterpret_cast<float2*>(hto + r * VD + 8 * nt + 2 * cq) = make_float2(st[mt][nt][2 * hh], st[mt][nt][2 * hh + 1]);
                }
    }
#endif
}

void gdnh75_fwd
(
    at::Tensor k, at::Tensor w, at::Tensor u, at::Tensor g,
    c10::optional<at::Tensor> h0, at::Tensor h,
    c10::optional<at::Tensor> v_new, c10::optional<at::Tensor> ht
)
{
    const at::cuda::OptionalCUDAGuard guard(k.device());
    TORCH_CHECK(k.dtype() == at::kHalf && w.dtype() == at::kHalf && u.dtype() == at::kHalf && h.dtype() == at::kHalf, "fp16 k/w/u/h");
    TORCH_CHECK(g.dtype() == at::kFloat, "fp32 g");
    TORCH_CHECK(k.is_contiguous() && w.is_contiguous() && u.is_contiguous() && g.is_contiguous() && h.is_contiguous(), "contiguous");
    int B = k.size(0), T = k.size(1), H = k.size(2), HV = u.size(2);
    TORCH_CHECK(k.size(3) == KD && w.size(3) == KD && u.size(3) == VD, "K = V = 128 only");
    TORCH_CHECK(w.size(2) == HV && HV % H == 0, "head counts");
    const float* h0p = nullptr;
    if (h0.has_value()) { TORCH_CHECK(h0->dtype() == at::kFloat && h0->is_contiguous()); h0p = h0->data_ptr<float>(); }
    half* vnp = nullptr;
    if (v_new.has_value()) { TORCH_CHECK(v_new->dtype() == at::kHalf && v_new->is_contiguous()); vnp = (half*) v_new->data_ptr(); }
    float* htp = nullptr;
    if (ht.has_value()) { TORCH_CHECK(ht->dtype() == at::kFloat && ht->is_contiguous()); htp = ht->data_ptr<float>(); }
    dim3 grid(VD / BV, B * HV);
    gdnh75_kernel<<<grid, NTHREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const half*) k.data_ptr(), (const half*) w.data_ptr(), (const half*) u.data_ptr(), g.data_ptr<float>(),
        h0p, (half*) h.data_ptr(), vnp, htp, T, H, HV);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#else

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

void gdnh75_fwd(at::Tensor k, at::Tensor w, at::Tensor u, at::Tensor g, c10::optional<at::Tensor> h0, at::Tensor h, c10::optional<at::Tensor> v_new, c10::optional<at::Tensor> ht)
{
    TORCH_CHECK(false, "Turing (sm_75) kernel not available on ROCm");
}

#endif
