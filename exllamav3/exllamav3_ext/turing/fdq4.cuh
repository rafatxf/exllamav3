#pragma once

#include <ATen/Tensor.h>

// Flash-decoding over a packed 4-bit exllamav3 cache on sm_75 (eager path; the CUDA-graph path compiles
// fdq4_bc.cu.in per slot shape). q (bsz, q_len, n_q_heads, 256) fp16 contiguous; qk/qv (pages, 256,
// n_kv_heads * 32) int32 and sk/sv (pages, 256, n_kv_heads * 8) fp16 as held by CacheLayer_quant with 4-bit
// K and V; block_table (bsz, pages_per_seq) and cache_seqlens (bsz) int32, the new rows already appended
// (pre_appended = q_len). At most 48 q rows per kv head (q_len * group). num_splits <= 0 picks a split
// count that fills the device; hb_n = 1
void fdq4_decode
(
    const at::Tensor& q,
    const at::Tensor& qk,
    const at::Tensor& sk,
    const at::Tensor& qv,
    const at::Tensor& sv,
    const at::Tensor& block_table,
    const at::Tensor& cache_seqlens,
    at::Tensor& out,
    int max_kv_len,
    int pre_appended,
    double sm_scale,
    int num_splits,
    int hb_n
);

// Co-resident split blocks per SM for nt 8-row n-tiles (sizes the split count of graph slots)
int fdq4_blocks_per_sm(int nt);
