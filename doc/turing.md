# Turing (sm_75) fast paths

ExLlamaV3 runs on Turing (RTX 20xx, including the popular 22 GB RTX 2080 Ti mods) since v1.5.2, but several
generic code paths leave most of these chips idle. Four facts about GeForce Turing drive everything below:

- **No bf16 tensor cores.**
- **fp32-accumulate HMMA runs at half the fp16-accumulate rate:** measured 507 vs 987 FLOP/clk/SM on a 2080 Ti.
- **A block gets 64 KB of shared memory,** so the Triton attention tile ladders shrink to a few percent of tensor peak.
- **Triton lowers `tl.dot` to scalar FMA on sm_75.** Nsight Compute shows zero tensor instructions, so every Triton kernel with matrix products runs on the CUDA cores.

This document describes the paths added for sm_75, the results on two model families (Qwen3.5 / 3.6 / 3.8 with
Gated DeltaNet, and Gemma 4 with sliding-window attention and MoE), and how to reproduce them. On sm_75 every path
is **on by default** and picks itself by layer shape; every other architecture keeps upstream behaviour unless a
switch is set explicitly.

- [Setup: reproducing these numbers](#setup-reproducing-these-numbers)
- [What is included](#what-is-included)
- [Results: Qwen3.8-27B](#results-qwen38-27b)
- [Results: Gemma 4 26B-A4B](#results-gemma-4-26b-a4b)
- [Accuracy](#accuracy)
- [Troubleshooting](#troubleshooting)

## Setup: reproducing these numbers

Everything here was measured on one RTX 2080 Ti 22 GB in a headless Linux box, serving through TabbyAPI.

### 1. System

| Component | Tested with | Notes |
|---|---|---|
| GPU | RTX 2080 Ti 22 GB (sm_75) | Any sm_75 card gets the same paths; 22 GB is what fits these models at 192K-256K context |
| OS / driver | Ubuntu 26.04, NVIDIA 595.91 | |
| CUDA toolkit | 13.0 | **`nvcc` is needed at runtime**, not only to build: the attention kernels compile one small cubin per layer shape on first use (cached in `~/.cache/exllamav3/fdq4`). Set `CUDA_HOME` or put `nvcc` on `PATH` |
| Python / PyTorch | 3.12 / 2.9.1+cu130 | |

### 2. Build

```sh
git clone -b turing https://github.com/rafatxf/exllamav3
cd exllamav3
python -m venv ~/exl3-venv && . ~/exl3-venv/bin/activate
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install ninja
export CUDA_HOME=/usr/local/cuda-13.0 TORCH_CUDA_ARCH_LIST=7.5
pip install --no-build-isolation -e .           # ~12 min on a 6-core CPU
python -c "from exllamav3.ext import exllamav3_ext as e; print('sm_75 kernels:', hasattr(e, 'fa75_fwd_win'))"
```

### 3. Models

| Model | Target | Draft (speculative decoding) |
|---|---|---|
| Qwen3.8-27B | [`turboderp/Qwen3.8-27B-exl3`](https://huggingface.co/turboderp/Qwen3.8-27B-exl3), branch `4.0bpw` | [`r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw`](https://huggingface.co/r0b0tlab/Qwen3.8-27B-DFlash2-EXL3-4.00bpw) (or `convert.py -b 4.0` on [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)) |
| Gemma 4 26B-A4B | [`turboderp/gemma-4-26B-A4B-it-exl3`](https://huggingface.co/turboderp/gemma-4-26B-A4B-it-exl3), branch `4.10bpw` | [`z-lab/gemma-4-26B-A4B-it-DFlash`](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash), bf16 as published |

```sh
hf download turboderp/gemma-4-26B-A4B-it-exl3 --revision 4.10bpw --local-dir ~/models/gemma-4-26B-A4B-it-exl3-4.10bpw
hf download z-lab/gemma-4-26B-A4B-it-DFlash --local-dir ~/models/gemma-4-26B-A4B-it-DFlash
```

Notes:
- **Gemma DFlash drafter:** the z-lab checkpoint does not pin its `tap_shift`; this branch recognizes it and uses 0
  (3.2 instead of 0.8 accepted tokens per round). An EXL3 4.0 bpw conversion of it was measured 4-10% slower than bf16
  at 4K+ context (lower acceptance), so the bf16 checkpoint is the one to use.
- **Other bitrates:** fewer bits do not decode faster on Turing (the EXL3 trellis decode is compute-bound there); pick
  the bitrate for quality and VRAM.

### 4. TabbyAPI

[TabbyAPI](https://github.com/theroyallab/tabbyAPI) accepts compute capability 7.5 since theroyallab/tabbyAPI#463.
Install it into the same virtual environment (so it imports this ExLlamaV3), and optionally apply the patches in
[`contrib/tabbyapi/`](../contrib/tabbyapi/) (made against TabbyAPI `816c321`):

```sh
git clone https://github.com/theroyallab/tabbyAPI && cd tabbyAPI
git checkout 816c321 && git am ../exllamav3/contrib/tabbyapi/*.patch     # optional, see contrib/tabbyapi/README.md
pip install -e .                                                           # into ~/exl3-venv
```

Run it with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (needed for ~190K-token prompts next to a draft
model's cache). The two configurations used for the results below (`config.yml`, relevant parts):

<details>
<summary>Qwen3.8-27B + DFlash2, 192K context</summary>

```yaml
model:
  model_dir: /home/you/models
  model_name: Qwen3.8-27B-exl3-4.0bpw
  max_seq_len: 196608
  cache_size: 196608
  cache_mode: Q4
  chunk_size: 2048
  max_batch_size: 1
  reasoning: true
  reasoning_start_token: "<think>"
  reasoning_end_token: "</think>"
  tool_format: qwen3_coder
draft_model:
  draft_mode: model
  draft_model_dir: /home/you/models
  draft_model_name: Qwen3.8-27B-DFlash2-exl3-4.0bpw
  draft_cache_mode: Q4
  draft_num_tokens: 7
sampling:
  override_preset: qwen38        # temperature 1.0, top_k 20, top_p 0.95 (contrib/tabbyapi patch 0002)
```
</details>

<details>
<summary>Gemma 4 26B-A4B + DFlash, 256K context</summary>

```yaml
model:
  model_dir: /home/you/models
  model_name: gemma-4-26B-A4B-it-exl3-4.10bpw
  max_seq_len: 262144
  cache_size: 262144
  cache_mode: Q4
  chunk_size: 2048
  max_batch_size: 1
  reasoning: true                # tags detected from the chat template (<|channel>thought ... <channel|>)
  tool_format: gemma4
draft_model:
  draft_mode: model
  draft_model_dir: /home/you/models
  draft_model_name: gemma-4-26B-A4B-it-DFlash
  draft_cache_mode: Q4
  draft_num_tokens: 7
sampling:
  override_preset: gemma4        # temperature 1.0, top_k 64, top_p 0.95 (contrib/tabbyapi patch 0002)
```
18.9 GB of VRAM with the draft model. Thinking is off by default in Gemma's template; clients turn it on with
`chat_template_kwargs: {"enable_thinking": true}`.
</details>

Keep `output_chunking` at its default (on). Without it a request with no `max_tokens` reserves the whole cache, and
concurrent requests wait.

**Sampler defaults matter for speed.** When a client sends no sampling parameters, TabbyAPI samples at temperature
1.0 over the whole vocabulary (top-k 0, top-p 1.0). Besides hurting quality, that halves the drafts' acceptance. Use
the model's `generation_config.json` values as the server defaults (the presets above).

### 5. GPU settings (optional)

The results were taken at a 240 W power limit with the core clock capped at 1650 MHz and a +240 MHz V/F offset:

```sh
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 240
sudo nvidia-smi -lgc 300,1650
# +240 MHz GPC V/F offset through NVML (nvmlDeviceSetGpcClkVfOffset, root): silicon-specific, validate with a burn-in
```

Decode on Turing is power-capped most of the time. At stock clocks expect roughly 10-15% less decode and prefill.
Undervolting is not required for any of the kernels.

### 6. What to expect

See the results sections. In short, through TabbyAPI:
- **Qwen3.8-27B + DFlash2:** 82 t/s decode on short prompts, 58-60 t/s at 64K-188K; prefill 1100 / 890 / 580 t/s at 16K / 64K / 188K.
- **Gemma 4 26B-A4B + DFlash:** 130-280 t/s on code and reasoning (acceptance-dependent), ~95-105 t/s on prose;
  without a draft 98 t/s short, 86 t/s at 64K, 70 t/s at 224K; prefill 2200 / 1710 / 720 t/s at 16K / 64K / 224K.

## What is included

| Path | Switch | Default on sm_75 | What it does |
|---|---|---|---|
| Prefill attention on a dequantized window | `EXL3_SDPA_PREFILL` | on | Prefill chunks (q_len > 8) with a quantized cache dequantize only the referenced window and attend with PyTorch SDPA (per KV group) or fa75, instead of the Triton paged-prefill kernels |
| fa75 | `EXL3_FA75` | on | Flash-attention prefill kernel for head_dim 256 on HMMA: 45–47 TFLOPS vs 10–14 for the cutlass SDPA kernel |
| fa75 sliding window | `EXL3_FA75_SWA` | on | Causal-window variant (`fa75_fwd_win`) for SlidingAttention prefill (Gemma 4): the chunk attends over [hot state ‖ new K/V] instead of the Triton paged-prefill kernel (~26 ms per 2048-token chunk and layer): 32K prefill 24.4 → 14.2 s |
| fdq4 | `EXL3_FDQ4` | on | Flash-decoding straight from 4-bit K/V caches, eager and CUDA-graph paths: ×8–11 at 128K vs the Triton decode kernel; fast Walsh-Hadamard q rotation and per-page addressing: 1 token 121 / 273 / 470 → 87 / 229 / 416 µs at 16K / 64K / 128K, 8 tokens (verify) −5 to −17%. The graph-path cubins are compiled per shape and cover head_dim 256 and 512, sliding windows, and more than 48 q rows per kv head (row split) |
| fd16 | `EXL3_FD16` | on | The same split / softmax / combine structure over fp16 K/V: SlidingAttention's state ring (Gemma 4, where the Triton decode kernel ran at ~46 GB/s) and fp16 caches |
| Draft-block attention | `EXL3_FDQ4_BLOCK` | on | DFlash drafters run a 16-token block every round; its attention over the 4-bit draft cache goes through the fdq4 cubins (head_dim 128, non-causal on full-attention layers, windowed on sliding ones), launched through the CUDA driver, instead of dequantizing the whole context for SDPA: drafter round 24.3 → 22.3 ms at 1.5K, 39.5 → 24.8 ms at 30K |
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
| Sliding-window decode | always | — | The Triton flash-decoding kernel splits only the window across its programs, instead of splitting the whole sequence and masking (which left a short window to one or two programs): DFlash2 drafter at 64K 520 → 140 µs per layer (any architecture) |
| DFlash2 speculative sampling | `EXL3_DFLASH_SPEC`, `EXL3_DFLASH_SPEC_TSCALE` | on (any architecture) | With temperature > 0 and a stateless sampler, sampled drafts verified by rejection sampling (lossless) instead of match-the-sample: acceptance on the drafter card's tasks at T = 1.0 4.19 → 4.51 on average; see `doc/env_vars.md`. DFlash (v1) drafters have no candidate selector and keep match-the-sample |
| DFlash drafter head on the used positions | always | — | A DFlash (v1) drafter's output head (the target's, 262K vocab on Gemma 4) runs on the positions the round uses instead of the whole 16-token block |
| DFlash tap_shift table | always | — | Released checkpoints that don't pin `tap_shift` are looked up by shape (z-lab's Gemma 4 26B-A4B drafter: 0, ~4× the acceptance of the default) |
| Inline recurrent checkpoint | `EXL3_INLINE_RECURRENT_CHECKPOINT` | on (any architecture) | Recurrent models take the prompt's last-page checkpoint inside the prefill chunk instead of a separate pass that reconstructs every weight for < 256 rows: +21% at 1K, +10% at 2K |
| Fused reconstruct occupancy | always | — | The fused reconstruct + Hadamard kernel reads packed tiles from global memory on sm_75: two blocks per SM, −14% kernel time |
| Unfused multi-projection GEMMs | `EXL3_MGEMM` | 0 (unfused) | Lets every projection take the GEMV path (the fused kernel has none): ~10% faster single-token decode at 240 W; `EXL3_MGEMM=1` keeps the fused kernels |

Setting a switch to `0` disables that path, which is useful for A/B tests. Python-side switches are read on every call; C++-side ones are read once.

Generic fixes that came out of this work (any architecture): requeued jobs count all their tokens, drafting jobs
requeue on a page boundary (no re-prefill of up to 2048 tokens per segment), and the speculative block stops where a
grammar filter activates.

### Which models benefit

- **Every EXL3 model on Turing:** fp16-accumulate GEMM, the GEMV rule, fp16-accumulate reconstruct GEMMs and SDPA prefill (with a quantized cache).
- **head_dim 256 attention:** fa75 (with or without a sliding window).
- **head_dim 256 / 512 attention with a 4-bit K/V cache:** fdq4 (decode and draft verify), windowed or not.
- **Sliding-window layers with an fp16 state (SlidingAttention):** fd16 and windowed fa75.
- **DFlash drafters with a 4-bit draft cache:** the draft-block path (head_dim 128 / 256 / 512).
- **Gated DeltaNet models:** the GDN paths (K = V = 128 for gdnh75).

The Qwen3.5 / 3.6 / 3.8 family and Gemma 4 26B-A4B match these conditions. Unsupported shapes fall through to the
upstream kernels.

## Results: Qwen3.8-27B

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
| 4K | 55.8 ms (17.9 t/s) | **30.0 ms (33.3 t/s)** | ×1.86 |
| 64K | 84.2 ms (11.9 t/s) | **32.7 ms (30.6 t/s)** | ×2.57 |
| 128K | — | **35.7 ms (28.0 t/s)** | — |

Contributions, measured step by step at a 175 W limit (ms per token at 4K / 64K):

| Step | 4K | 64K |
|---|---|---|
| fdq4 decode attention | 58.7 | 63.0 |
| + fp16-accumulate EXL3 GEMM | 53.3 | 58.7 |
| + GEMV rule | 41.9 | 47.8 |
| + unfused multi-projections | 41.0 | 45.1 |

At 240 W, the split-k GEMV takes decode from 34.3 / 37.3 to 30.6 / 33.7 ms per token at 4K / 64K (ABBA); the GDN
recurrent kernel and the fdq4 changes bring it to 30.0 / 32.7. A DFlash2 iteration (7 draft tokens, verify at m = 8,
engine, short context) goes from 46.9 to 40.2 ms (split-k GEMV, GDN recurrent kernel with lazy history), and from
50.6 to 47.4 ms at 64K (fdq4, windowed drafter attention).

### End to end (TabbyAPI, 240 W power limit, core clock capped at 1650 MHz with a +240 MHz V/F offset)

| Context | Prefill t/s | Decode t/s, DFlash2 draft | Decode t/s, MTP-3 draft |
|---|---|---|---|
| Short prompts | — | **81.8** (was 66.5) | 52.7 |
| 16K | **1103** (was 986) | **70.2** (was 58.7) | — |
| 64K | **891** (was 809) | **60.1** (was 45.3) | — |
| 128K | **692** (was 638) | **58.9** (was 49.7) | — |
| 188K | **579** (was 539) | **57.6** (was 54.3) | 48.7 |
| 244K | 442 | — | 43.4 |

Bold: current branch (decode phase), greedy, 384 new tokens, one pass (the earlier column: 512 tokens, ABBA). With
a draft model the decode rate also follows the acceptance of the generated text, which changes with any numeric
change under greedy decoding, so single rows move by ±10%; the suite means are the comparison. MTP rows are from
the previous build.

Sampled decoding (temperature 1.0, top-p 0.95, top-k 20, same 16 prompts, DFlash2): 74.4 → 79.1 t/s mean with
speculative sampling (`EXL3_DFLASH_SPEC`), geometric-mean ratio ×1.07; single prompts vary with the sampled text.

Without fdq4, MTP on the same card decodes 31 / 17 / 12 / 7.5 t/s at 16K / 64K / 128K / 244K (v1.5.2). Needle-in-a-haystack: 25/25 at 32K, 128K, 188K and 250K; 15/15 at 64K, 128K and 188K with the current branch (DFlash2).

Short-prompt suites use 16 prompts × 2 rounds in ABBA order, with bootstrap confidence intervals and paired per-prompt ratios.

## Results: Gemma 4 26B-A4B

Same card and settings; [`turboderp/gemma-4-26B-A4B-it-exl3`](https://huggingface.co/turboderp/gemma-4-26B-A4B-it-exl3)
4.10 bpw (MoE: 128 experts, 8 active, ~4B active parameters; 25 sliding-window layers with a 1024-token window and
5 global layers with head_dim 512), Q4 cache, 256K context. "Before" is this branch before the Gemma work (the Qwen
paths already in place: split-k GEMV etc.), i.e. every Gemma attention layer on the Triton kernels.

### Decode without a draft model (engine, greedy)

| Context | Before | **This branch** |
|---|---|---|
| Short | 83.2 t/s | **98.1 t/s** |
| 4K | 63.2 t/s | **92.8 t/s** |
| 16K | 55.8 t/s | **91.1 t/s** |
| 64K | 41.1 t/s | **86.4 t/s** |

Step by step: fdq4 on the global layers (head_dim 512) takes 64K from 41.1 to 63.4 t/s; fd16 on the 25 sliding
layers (fp16 state ring) takes it to 86.4.

### Prefill (TabbyAPI, no draft model)

| Prompt length | Before | **This branch** |
|---|---|---|
| 16K | 1496 t/s | **2199 t/s** |
| 64K | 1115 t/s | **1712 t/s** |
| 128K | 818 t/s | **1108 t/s** |
| 224K | 573 t/s | **722 t/s** |

At 32K the sliding-window layers went from 10.5 s (Triton paged prefill) to 0.35 s (windowed fa75); the global
layers' attention (PyTorch cutlass SDPA, head_dim 512, ~16 TFLOPS) is now 39% of the prefill.

### Decode with the z-lab DFlash drafter (7 draft tokens, engine, greedy)

| Task | No draft | **DFlash, thinking on** | **DFlash, thinking off** |
|---|---|---|---|
| Short prompts, mean of 4 | 97-98 t/s | **158-167 t/s** | **133 t/s** |
| Math, short | | 262-280 t/s | 232 t/s |
| Code, short | | 165-173 t/s | 185 t/s |
| Prose (Spanish / English), short | | 114-131 t/s | 93-106 t/s |
| Math at 16K / 32K context | ~91 / ~89 t/s | **248-254 / 228-236 t/s** | |
| Code at 16K context | 91 t/s | | **152 t/s** |
| Summarizing 30K / 60K of prose | ~88 / ~86 t/s | | 88 / 73 t/s |

The drafter accepts 3.5-6 tokens per round on code and math and ~2 on prose, while a round (verify 8 positions of a
MoE: ~40 distinct experts instead of 8, plus the drafter) costs 22-26 ms against 10-12 ms per plain token. Speculation
therefore wins on code and reasoning and roughly breaks even or loses on long prose.

Through TabbyAPI (DFlash): a thinking chat request decodes at 226-232 t/s, tool calls and JSON schemas (with and
without thinking) work, and needles are 6/6 at 64K and 128K. Without a draft model: 98 / 90 / 86 / 79 / 70 t/s at
short / 16K / 64K / 128K / 224K, needles 9/9 at 64K, 128K and 224K.

## Accuracy

The GEMM and GDN changes alter numerics; the attention kernels match their references to fp16 rounding.

**Logits at 2K context** (Qwen3.8-27B, 16 sequences × 256 positions, each path vs the same build without it):

| Path | KL | Perplexity change |
|---|---|---|
| fp16-accumulate EXL3 GEMM | 8.3e-5 | — |
| GEMV vs GEMM | 7.7e-5 | — |
| GDN fp16 + cuBLAS output stage | 5.1e-5 | — |
| `EXL3_HGEMM_F16=1` | 7.4e-4 | +0.02% |
| `EXL3_HGEMM_F16=2` | 1.3e-3 | +0.08% |

For scale, going from 4.0 to 3.5 bpw costs +3.8% perplexity on the same text.

**Long context through the real cached prefill path** (Qwen3.8-27B, paged Q4 cache, chunked prefill, GDN state carried across chunks), 32K tokens:

| Configuration | PPL | 0–2K | 2–8K | 8–32K |
|---|---|---|---|---|
| v1.5.3 | 7.7167 | 9.803 | 8.753 | 7.330 |
| This branch | 7.7207 (+0.05%) | 9.784 | 8.749 | 7.337 |

**Decode path** (2K tokens fed 1 or 8 at a time, so every projection runs through the GEMV kernels), split-k GEMV
vs the block-per-group GEMV: PPL 9.8003 → 9.7881 (1 at a time) and 9.7983 → 9.8026 (8 at a time), top-1 agreement
100%. The GDN recurrent kernel and its lazy history are bit-identical to the generic kernel.

**Gemma 4 attention kernels:**
- **fdq4 / fd16 against fp32 attention** over a shuffled paged cache (head_dim 256 / 512, q_len 1 / 8, window 1023 or none, lengths 3 to 2800 around window and chunk edges): worst relative error ≤ 1.3e-3.
- **fa75 with a window against fp32 attention** (windows 5 to 1023, lengths that are not multiples of the tile): worst relative error 5.4e-4.
- **Teacher-forced decode logits vs the Triton path** over 160 tokens at 4K and 32K: argmax agreement 99.4-100%, mean KL 3e-4 to 2e-3. For scale, switching the cache from Q4 to Q8 alone gives a mean KL of 1.3e-2 on the same sequence.
- **Draft-block path:** changes only the drafter (acceptance unchanged); the target's logits are bit-identical.

The difference does not grow with context. With every switch off, this branch reproduces its base bit for bit (KL 0).

## Troubleshooting

- **`!! fdq4: graph-path kernels unavailable`**: `nvcc` was not found. Set `CUDA_HOME`. Decode still works, on the slower Triton kernels.
- **The first requests after a restart are a few seconds slower**: each new attention shape compiles its cubin once (cached on disk afterwards).
- **Low draft acceptance with a DFlash drafter** (a fraction of a token per round): check its `tap_shift`. This branch knows the z-lab Gemma 4 26B-A4B drafter; for other checkpoints try `"tap_shift": 0` or `1` under `dflash_config` in the drafter's `config.json`.
- **Gemma 4 produces garbage on raw text**: the instruction model needs its chat template (`<|turn>user ... <turn|>`); raw-text continuation degenerates on every backend path.
- **TabbyAPI: JSON schema + Gemma thinking returns broken JSON**, or **"Job requires N pages (only N-1 available)"** with `output_chunking: false` and a draft model: apply the patches in `contrib/tabbyapi/`.
- **Tests:** `tests/test_turing_*.py` (fa75, fa75 window, fdq4, fd16, gdnh75, gdnwy75, gdno75, gemv_sk, gdn_rec75), `tests/test_sm75_gemm.py`, `tests/test_triton_decode_window.py`, `tests/test_dflash_spec_sampling.py`, `tests/test_dflash2_walk_sample.py`, `tests/turing_inline_checkpoint_check.py`.
- **Kernels** live in `exllamav3_ext/turing/`; they compile for any sm_75+ target and are inert on ROCm.

## Credits

This work was developed with extensive help from an AI coding assistant (Anthropic's Claude), including the CUDA kernels, the analysis and the benchmarks. Every number above was measured on real hardware with the methodology described.
