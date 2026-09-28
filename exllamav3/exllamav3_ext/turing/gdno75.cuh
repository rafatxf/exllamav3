#pragma once

#include <ATen/Tensor.h>

// Gated delta rule output stage on sm_75 tensor cores (FLA's chunk_fwd_kernel_o), per 64-token chunk and value head:
// o = scale * [(q @ h) * exp2(g_i) + tril_incl((q @ k^T) * exp2(g_i - g_j)) @ v]. K = V = 128, fp16 q/k [B,T,H,K],
// v [B,T,HV,V], h [B,NT,HV,K,V], fp32 g [B,T,HV] (chunk-local cumsum, log2 domain), fp16 o [B,T,HV,V]
void gdno75_fwd
(
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor h,
    at::Tensor g,
    at::Tensor o,
    double scale
);
