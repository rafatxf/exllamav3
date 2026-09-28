# Turing (sm_75) fast paths

ExLlamaV3 runs on Turing (RTX 20xx, including the popular 22 GB RTX 2080 Ti mods) since v1.5.2, but several
generic code paths leave most of these chips idle. Four facts about GeForce Turing drive everything below:

- **No bf16 tensor cores.**
- **fp32-accumulate HMMA runs at half the fp16-accumulate rate:** measured 507 vs 987 FLOP/clk/SM on a 2080 Ti.
- **A block gets 64 KB of shared memory,** so the Triton attention tile ladders shrink to a few percent of tensor peak.
- **Triton lowers `tl.dot` to scalar FMA on sm_75.** Nsight Compute shows zero tensor instructions, so every Triton kernel with matrix products runs on the CUDA cores.

This document describes the paths added for sm_75. On sm_75 each one is **on by default**. Every other architecture keeps upstream behaviour unless the switch is set explicitly.

## What is included

| Path | Switch | Default on sm_75 | What it does |
|---|---|---|---|
| Prefill attention on a dequantized window | `EXL3_SDPA_PREFILL` | on | Prefill chunks (q_len > 8) with a quantized cache dequantize only the referenced window and attend with PyTorch SDPA (per KV group) or fa75, instead of the Triton paged-prefill kernels |
| fa75 | `EXL3_FA75` | on | Flash-attention prefill kernel for head_dim 256 on HMMA: 45–47 TFLOPS vs 10–14 for the cutlass SDPA kernel |
| fdq4 | `EXL3_FDQ4` | on | Flash-decoding straight from 4-bit K/V caches, eager and CUDA-graph paths: ×8–11 at 128K vs the Triton decode kernel |
| fp16-accumulate EXL3 GEMM | build-time (`-DEXL3_NO_H_ACC_SM75` disables) | on | The existing sm_86 H_ACC path, enabled for sm_75 |
| GEMV dispatch rule | always (`EXL3_GEMV=0` disables GEMV) | on | Decode-shape GEMMs take the GEMV kernels: weight time 49.7 → 34.1 ms per token |
| Split-k GEMV | `EXL3_GEMV_SK` | on | 1 ≤ m ≤ 8 GEMV with equal work per SM for any shape (no partial last wave, no idle SMs), deterministic reduction, output transform in registers; also covers 5–8 bpw heads (6 bpw head: 387 → 555 GB/s). Weight time per token 30.4 → 26.6 ms at m = 1, 32.1 → 30.1 ms at m = 8 |
| fp16-accumulate reconstruct GEMM | `EXL3_HGEMM_F16` (0/1/2) | 2 | Prefill GEMMs through cuBLAS `CUBLAS_COMPUTE_16F` (h1688 kernels at full rate); level 2 also covers fp32-output GEMMs |
| GDN fp16 operands | `EXL3_GDN_FP16` | on | Gated delta rule chunk kernels on fp16 instead of bf16 |
| gdno75 | `EXL3_GDN_O_CUDA` | on | Gated delta rule output stage (`chunk_fwd_o`) on HMMA: ×73 vs the Triton kernel (0.29 TFLOPS, spills), ×9 vs the cuBLAS form |
| GDN output stage on cuBLAS | `EXL3_GDN_O_TORCH` | on | Fallback when gdno75 does not apply: `chunk_fwd_o` as batched GEMMs, ×8.4 |
| gdnwy75 | `EXL3_GDN_WY_CUDA` | on | Gated delta rule WY representation (kkt + solve_tril + recompute_w_u) on HMMA: ×4 vs the Triton kernels |
| gdnh75 | `EXL3_GDN_H_CUDA` | on | Gated delta rule state recurrence on HMMA: ×10 vs the Triton kernel |
| GDN recurrent step | `EXL3_GDN_REC75` | on | Decode / verify state update with the state in registers (×1.9 at one token) and lazy speculative history: the verify keeps the initial state and the step inputs instead of every intermediate state (3.1 MB each), and a rewind replays the accepted steps. Bit-identical outputs and rewound states; DFlash2 iteration 42.7 → 40.2 ms |
| Sliding-window decode skip | always | — | The Triton decode kernel starts at the window instead of masking the whole sequence (any architecture) |
| Inline recurrent checkpoint | `EXL3_INLINE_RECURRENT_CHECKPOINT` | on (any architecture) | Recurrent models take the prompt's last-page checkpoint inside the prefill chunk instead of a separate pass that reconstructs every weight for < 256 rows: +21% at 1K, +10% at 2K |
| Fused reconstruct occupancy | always | — | The fused reconstruct + Hadamard kernel reads packed tiles from global memory on sm_75: two blocks per SM, −14% kernel time |
| Unfused multi-projection GEMMs | `EXL3_MGEMM` | 0 (unfused) | Lets every projection take the GEMV path (the fused kernel has none): ~10% faster single-token decode at 240 W; `EXL3_MGEMM=1` keeps the fused kernels |

Setting a switch to `0` disables that path, which is useful for A/B tests. Python-side switches are read on every call; C++-side ones are read once.

### Which models benefit

- **Every EXL3 model on Turing:** fp16-accumulate GEMM, the GEMV rule, fp16-accumulate reconstruct GEMMs and SDPA prefill (with a quantized cache).
- **head_dim 256 attention:** fa75.
- **head_dim 256 attention with a 4-bit K/V cache and GQA group × q_len ≤ 48:** fdq4.
- **Gated DeltaNet models:** the GDN paths (K = V = 128 for gdnh75).

The Qwen3.5 / 3.6 / 3.8 family matches every condition. Unsupported shapes fall through to the upstream kernels.

## Results

RTX 2080 Ti 22 GB, Qwen3.8-27B EXL3 4.0 bpw, Q4 cache. Unless noted, the card runs at a 240 W power limit with
the core clock capped at 1650 MHz and a +240 MHz V/F offset. The baseline is v1.5.3, which already includes the
sm_75 prefill tiles and FLA autotune configs from #411.

### Prefill (engine, tokens/s, mean of two ABBA passes)

| Prompt length | v1.5.3 | **This branch** | Speed-up |
|---|---|---|---|
| 2K | 588 | **1249** | ×2.1 |
| 16K | 267 | **1189** | ×4.5 |
| 64K | 93 | **920** | ×9.9 |

What each part contributes, measured by switching it back to the v1.5.3 path within this branch:

| Configuration | 2K | 16K | 64K |
|---|---|---|---|
| Attention back on the Triton paged-prefill kernels (`EXL3_SDPA_PREFILL=0`) | 885 | 310 | 97 |
| Gated delta rule back on FLA/Triton (`EXL3_GDN_*=0`) | 1034 | 1000 | 807 |

The attention path carries most of the long-context gain; the GDN kernels add 14-21% on top of the #411 configs.
A larger prefill chunk (`max_chunk_size` / TabbyAPI `chunk_size` 8192 instead of 2048) amortizes the weight
reconstruct over more rows: 1249 / 966 t/s at 16K / 64K, at the cost of larger activation buffers.

Where the time goes at 240 W: up to 16K the cuBLAS GEMMs take 65-70% and run at ~92% of the fp16 tensor peak for
the clock the power limit allows (~1550-1610 MHz), the weight reconstruct 12-13%; at 64K the GEMMs take 50% and fa75
30% (~80% of its fp32-accumulate peak).

### Decode (engine, no draft model)

| Context | v1.5.3 | **This branch** | Speed-up |
|---|---|---|---|
| 4K | 55.8 ms (17.9 t/s) | **30.6 ms (32.7 t/s)** | ×1.83 |
| 64K | 84.2 ms (11.9 t/s) | **33.7 ms (29.7 t/s)** | ×2.50 |

Contributions, measured step by step at a 175 W limit (ms per token at 4K / 64K):

| Step | 4K | 64K |
|---|---|---|
| fdq4 decode attention | 58.7 | 63.0 |
| + fp16-accumulate EXL3 GEMM | 53.3 | 58.7 |
| + GEMV rule | 41.9 | 47.8 |
| + unfused multi-projections | 41.0 | 45.1 |

At 240 W, the split-k GEMV takes decode from 34.3 / 37.3 to 30.6 / 33.7 ms per token at 4K / 64K (ABBA), and a
DFlash2 iteration (7 draft tokens, verify at m = 8) from 46.9 to 43.3 ms.

### End to end (TabbyAPI, 240 W power limit, core clock capped at 1650 MHz with a +240 MHz V/F offset)

| Context | Prefill t/s | Decode t/s, DFlash2 draft | Decode t/s, MTP-3 draft |
|---|---|---|---|
| Short prompts | — | 66.5 | 52.7 |
| 16K | 986 | 58.7 | — |
| 64K | 809 | 45.3 | — |
| 128K | 638 | 49.7 | — |
| 188K | 539 | 54.3 | 48.7 |
| 244K | 442 | — | 43.4 |

Without fdq4, MTP on the same card decodes 31 / 17 / 12 / 7.5 t/s at 16K / 64K / 128K / 244K (v1.5.2). Needle-in-a-haystack: 25/25 at 32K, 128K, 188K and 250K.

Short-prompt suites use 16 prompts × 2 rounds in ABBA order, with bootstrap confidence intervals and paired per-prompt ratios.

## Accuracy

The GEMM and GDN changes alter numerics; the attention kernels match their references to fp16 rounding.

**Logits at 2K context** (16 sequences × 256 positions, each path vs the same build without it):

| Path | KL | Perplexity change |
|---|---|---|
| fp16-accumulate EXL3 GEMM | 8.3e-5 | — |
| GEMV vs GEMM | 7.7e-5 | — |
| GDN fp16 + cuBLAS output stage | 5.1e-5 | — |
| `EXL3_HGEMM_F16=1` | 7.4e-4 | +0.02% |
| `EXL3_HGEMM_F16=2` | 1.3e-3 | +0.08% |

For scale, going from 4.0 to 3.5 bpw costs +3.8% perplexity on the same text.

**Long context through the real cached prefill path** (paged Q4 cache, chunked prefill, GDN state carried across chunks), 32K tokens:

| Configuration | PPL | 0–2K | 2–8K | 8–32K |
|---|---|---|---|---|
| v1.5.3 | 7.7167 | 9.803 | 8.753 | 7.330 |
| This branch | 7.7222 (+0.07%) | 9.798 | 8.753 | 7.337 |

The difference does not grow with context. With every switch off, this branch reproduces its base bit for bit (KL 0).

## Requirements and notes

- **Build from source** for sm_75 (`TORCH_CUDA_ARCH_LIST=7.5`). The kernels live in `exllamav3_ext/turing/`, compile for any sm_75+ target, and are inert on ROCm.
- **The fdq4 CUDA-graph path needs `nvcc` at runtime.** It compiles one small cubin per slot shape, cached under `~/.cache/exllamav3/fdq4`. Without nvcc it prints a one-time notice and the graph path keeps the Triton decode kernels; the eager fdq4 path does not need nvcc.
- **TabbyAPI** accepts compute capability 7.5 since theroyallab/tabbyAPI#463.
- **Tests:** `tests/test_turing_fa75.py`, `tests/test_turing_fdq4.py`, `tests/test_turing_gdnh75.py`.

## Credits

This work was developed with extensive help from an AI coding assistant (Anthropic's Claude), including the CUDA kernels, the analysis and the benchmarks. Every number above was measured on real hardware with the methodology described.
