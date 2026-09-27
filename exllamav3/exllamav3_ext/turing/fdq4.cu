// fdq4 eager path: flash-decoding over a packed 4-bit exllamav3 cache (see fdq4_core.cuh). The CUDA-graph
// path compiles fdq4_bc.cu.in per slot shape at runtime (modules/attention_fn/fdq4.py).
#if !defined(USE_ROCM)

#include <ATen/ATen.h>
#include <c10/util/Optional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "fdq4_core.cuh"

template <int NT>
__global__ void __launch_bounds__(FD_THREADS)
fdq4_split_kernel(
    const half* __restrict__ q, const uint32_t* __restrict__ qk, const half* __restrict__ sk,
    const uint32_t* __restrict__ qv, const half* __restrict__ sv,
    const int* __restrict__ block_table, const int* __restrict__ cache_seqlens,
    float* __restrict__ part_o, float* __restrict__ part_ml,
    int ql, int nq, int nkv, int pps, int split_len, int pre_appended, float scale_log2, int hb_n, int rb)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
    __trap();
#else
    const int kvh = blockIdx.y / hb_n, hb = blockIdx.y % hb_n;
    const int row0 = hb * rb, nrows = min(rb, ql * (nq / nkv) - row0);
    if (nrows <= 0) return;
    fdq4_split_body<NT>(blockIdx.x, kvh, blockIdx.z, gridDim.x, q, qk, sk, qv, sv, block_table, cache_seqlens,
                        part_o, part_ml, ql, nq, nkv, pps, split_len, pre_appended, scale_log2, row0, nrows);
#endif
}

__global__ void __launch_bounds__(FD_HD)
fdq4_combine_kernel(const float* __restrict__ part_o, const float* __restrict__ part_ml, half* __restrict__ out,
                    int splits, int ql, int nq, int nkv)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750)
    __trap();
#else
    fdq4_combine_body(blockIdx.x, blockIdx.y, blockIdx.z, part_o, part_ml, out, splits, ql, nq, nkv);
#endif
}

// q: (bsz, ql, nq, 256) fp16; qk/qv: (pages, 256, nkv*32) int32; sk/sv: (pages, 256, nkv*8) fp16
// block_table: (bsz, pps) int32; cache_seqlens: (bsz) int32 (tokens before the pre-appended ones)
void fdq4_decode(
    const at::Tensor& q, const at::Tensor& qk, const at::Tensor& sk, const at::Tensor& qv, const at::Tensor& sv,
    const at::Tensor& block_table, const at::Tensor& cache_seqlens, at::Tensor& out,
    int max_kv_len, int pre_appended, double sm_scale, int num_splits, int hb_n)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(q.dtype() == at::kHalf && q.is_contiguous() && q.size(3) == FD_HD, "q must be contiguous fp16 (b, ql, nq, 256)");
    TORCH_CHECK(qk.dtype() == at::kInt && qv.dtype() == at::kInt && sk.dtype() == at::kHalf && sv.dtype() == at::kHalf, "cache dtypes");
    TORCH_CHECK(qk.size(1) == FD_PAGE, "page size must be 256");
    TORCH_CHECK(block_table.dtype() == at::kInt && cache_seqlens.dtype() == at::kInt, "int32 tables");
    int bsz = q.size(0), ql = q.size(1), nq = q.size(2);
    int nkv = sk.size(2) / (FD_HD / 32);
    TORCH_CHECK(qk.size(2) == nkv * FD_HD / 8 && qv.size(2) == nkv * FD_HD / 8, "only 4-bit K/V supported");
    TORCH_CHECK(nq % nkv == 0, "nq % nkv");
    int R = ql * (nq / nkv);
    TORCH_CHECK(R <= FD_MAXR, "too many q rows per kv head");
    if (hb_n <= 0) hb_n = 1;
    int rb = (R + hb_n - 1) / hb_n;
    int NT = (rb + 7) / 8;
    int pps = block_table.size(1);

    if (num_splits <= 0)
    {
        int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        int per_sm = 1;
        #define OCC(N) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fdq4_split_kernel<N>, FD_THREADS, 0)
        switch (NT) { case 1: OCC(1); break; case 2: OCC(2); break; case 3: OCC(3); break;
                      case 4: OCC(4); break; case 5: OCC(5); break; default: OCC(6); break; }
        #undef OCC
        num_splits = std::max(1, (per_sm * sms) / (nkv * bsz * hb_n));
    }
    int max_len = std::min(pps * FD_PAGE, max_kv_len + pre_appended);
    int chunks = (max_len + FD_CHUNK - 1) / FD_CHUNK;
    num_splits = std::max(1, std::min(num_splits, chunks));
    int split_len = ((chunks + num_splits - 1) / num_splits) * FD_CHUNK;

    auto fopt = q.options().dtype(at::kFloat);
    at::Tensor part_o = at::empty({(long)bsz * nkv * num_splits * R * FD_HD}, fopt);
    at::Tensor part_ml = at::empty({(long)bsz * nkv * num_splits * R * 2}, fopt);
    float scale_log2 = (float)(sm_scale * 1.4426950408889634);

    dim3 grid(num_splits, nkv * hb_n, bsz);
    #define LAUNCH(N) fdq4_split_kernel<N><<<grid, FD_THREADS, 0, stream>>>( \
        (const half*) q.data_ptr(), (const uint32_t*) qk.data_ptr(), (const half*) sk.data_ptr(), \
        (const uint32_t*) qv.data_ptr(), (const half*) sv.data_ptr(), \
        (const int*) block_table.data_ptr(), (const int*) cache_seqlens.data_ptr(), \
        (float*) part_o.data_ptr(), (float*) part_ml.data_ptr(), \
        ql, nq, nkv, pps, split_len, pre_appended, scale_log2, hb_n, rb)
    switch (NT)
    {
        case 1: LAUNCH(1); break;
        case 2: LAUNCH(2); break;
        case 3: LAUNCH(3); break;
        case 4: LAUNCH(4); break;
        case 5: LAUNCH(5); break;
        case 6: LAUNCH(6); break;
        default: TORCH_CHECK(false, "unsupported row count");
    }
    #undef LAUNCH
    dim3 cgrid(R, nkv, bsz);
    fdq4_combine_kernel<<<cgrid, FD_HD, 0, stream>>>((const float*) part_o.data_ptr(), (const float*) part_ml.data_ptr(),
                                                    (half*) out.data_ptr(), num_splits, ql, nq, nkv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int fdq4_blocks_per_sm(int nt)
{
    int per_sm = 1;
    #define OCC(N) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fdq4_split_kernel<N>, FD_THREADS, 0)
    switch (nt) { case 1: OCC(1); break; case 2: OCC(2); break; case 3: OCC(3); break;
                  case 4: OCC(4); break; case 5: OCC(5); break; default: OCC(6); break; }
    #undef OCC
    return per_sm;
}

#else

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

void fdq4_decode(const at::Tensor& q, const at::Tensor& qk, const at::Tensor& sk, const at::Tensor& qv, const at::Tensor& sv, const at::Tensor& block_table, const at::Tensor& cache_seqlens, at::Tensor& out, int max_kv_len, int pre_appended, double sm_scale, int num_splits, int hb_n)
{
    TORCH_CHECK(false, "Turing (sm_75) kernel not available on ROCm");
}
int fdq4_blocks_per_sm(int nt)
{
    TORCH_CHECK(false, "Turing (sm_75) kernel not available on ROCm");
    return 0;
}

#endif
