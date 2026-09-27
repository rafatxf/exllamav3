#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

// Gated delta rule chunk-state recurrence on sm_75 tensor cores (the forward of FLA's
// chunk_gated_delta_rule_fwd_kernel_h). K = V = 128, chunk 64, fp16 k [B,T,H,K], w [B,T,HV,K], u [B,T,HV,V],
// fp32 g [B,T,HV] (chunk-local cumsum, log2 domain); writes h [B,NT,HV,K,V] (fp16), v_new (fp16) and the
// final state ht [B,HV,K,V] (fp32), starting from h0 (fp32) when given
void gdnh75_fwd
(
    at::Tensor k,
    at::Tensor w,
    at::Tensor u,
    at::Tensor g,
    c10::optional<at::Tensor> h0,
    at::Tensor h,
    c10::optional<at::Tensor> v_new,
    c10::optional<at::Tensor> ht
);
