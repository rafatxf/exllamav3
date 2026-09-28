#pragma once

#include <ATen/Tensor.h>

// Gated delta rule WY representation on sm_75 tensor cores (FLA's kkt + solve_tril + recompute_w_u): per 64-token
// chunk and value head, X = (I + strict_tril(K K^T exp2(g_i - g_j)) beta_i)^-1, u = X diag(beta) V and
// w = X diag(beta exp2(g)) K. K = V = 128, fp16 k [B,T,H,K], v [B,T,HV,V], beta [B,T,HV], fp32 g [B,T,HV]
// (chunk-local cumsum, log2 domain); fp16 w [B,T,HV,K], u [B,T,HV,V]
void gdnwy75_fwd
(
    at::Tensor k,
    at::Tensor v,
    at::Tensor beta,
    at::Tensor g,
    at::Tensor w,
    at::Tensor u
);
