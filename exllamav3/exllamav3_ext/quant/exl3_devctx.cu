#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "exl3_devctx.cuh"
#include "../util.h"
#include "../util.cuh"

//DevCtx::DevCtc()
//{
//    int num_sms[MAX_DEVICES] = {};
//    int cc[MAX_DEVICES] = {};
//    void* locks[MAX_DEVICES] = {};
//    std::mutex mtx;
//}

DevCtx& DevCtx::instance()
{
    static DevCtx ctx;
    return ctx;
}

int DevCtx::get_num_sms(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!num_sms[device])
        cuda_check(cudaDeviceGetAttribute(&num_sms[device], cudaDevAttrMultiProcessorCount, device));
    return num_sms[device];
}

int DevCtx::get_cc(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!cc[device])
    {
        cudaDeviceProp prop;
        cuda_check(cudaGetDeviceProperties(&prop, device));
        if (prop.major >= 10) cc[device] = CC_BLACKWELL;
        else if (prop.major >= 9) cc[device] = CC_HOPPER;
        else if (prop.major >= 8 && prop.minor >= 9) cc[device] = CC_ADA;
        else if (prop.major >= 8) cc[device] = CC_AMPERE;
        else cc[device] = CC_OLD;
    }
    return cc[device];
}

int DevCtx::get_smem_max(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!smem_max[device])
    {
        // The driver value, unclamped: this is a device capability, and callers that only need
        // "what may I request for an EXL3 kernel" clamp to SMEM_MAX themselves (see
        // get_smem_request). Clamping here would have made an sm_86 device indistinguishable
        // from a 90 KB one and dragged it onto the Turing code paths.
        int optin = 0;
        cuda_check(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        smem_max[device] = optin;
    }
    return smem_max[device];
}

int DevCtx::get_smem_request(int device)
{
    return MIN(get_smem_max(device), EXL3_SMEM_MAX_DEFAULT);
}

void* DevCtx::get_ws(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!ws[device])
    {
        c10::cuda::CUDAGuard guard(device);
        cudaError_t e = cudaMalloc(&ws[device], WORKSPACE_SIZE);
        TORCH_CHECK(e == cudaSuccess, "exl3 workspace allocation failed on device ", device, ": ", cudaGetErrorString(e));
    }
    return ws[device];
}

int* DevCtx::get_locks(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!locks[device])
    {
        c10::cuda::CUDAGuard guard(device);
        size_t size = (MAX_TILES_C + MAX_BARRIERS * 2 + MOE_SCHED_INTS) * sizeof(int);
        cudaError_t e = cudaMalloc(&locks[device], size);
        TORCH_CHECK(e == cudaSuccess, "exl3 lock buffer allocation failed on device ", device, ": ", cudaGetErrorString(e));
        e = cudaMemset(locks[device], 0, size);
        TORCH_CHECK(e == cudaSuccess, "exl3 lock buffer memset failed: ", cudaGetErrorString(e));
        if (!gemv_sk_ws[device])
        {
            int major = 0, minor = 0;
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
            if (major == 7 && minor == 5) alloc_gemv_sk_ws(device);
        }
    }
    return (int*) locks[device];
}

void* DevCtx::alloc_gemv_sk_ws(int device)
{
    c10::cuda::CUDAGuard guard(device);
    cudaError_t e = cudaMalloc(&gemv_sk_ws[device], EXL3_GEMV_SK_WS_FLOATS * sizeof(float));
    TORCH_CHECK(e == cudaSuccess, "exl3 split-k GEMV workspace allocation failed on device ", device, ": ", cudaGetErrorString(e));
    return gemv_sk_ws[device];
}

float* DevCtx::get_gemv_sk_ws(int device)
{
    std::lock_guard<std::mutex> lock(mtx);
    if (!gemv_sk_ws[device]) alloc_gemv_sk_ws(device);
    return (float*) gemv_sk_ws[device];
}

int g_get_cc(int device)
{
    return DevCtx::instance().get_cc(device);
}

int g_get_num_sms(int device)
{
    return DevCtx::instance().get_num_sms(device);
}

int g_get_smem_max(int device)
{
    return DevCtx::instance().get_smem_max(device);
}

void prepare_ctx(int device)
{
    DevCtx::instance().get_num_sms(device);
    DevCtx::instance().get_cc(device);
    DevCtx::instance().get_smem_max(device);
    DevCtx::instance().get_locks(device);
}
