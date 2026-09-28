"""
sm_75 split-k GEMV (exllamav3_ext/quant/exl3_gemv_sk_kernel.cuh), reached through exl3_gemm for 1 <= m <= 8 on
Turing: 2-4 bpw at n <= 12288, 5-8 bpw (output heads) at any n, including n large enough that one warp's range spans
several column groups. Checked against reconstruct-then-matmul (ground truth) and against the block-per-group
GEMV it replaces (EXL3_GEMV_SK=0, read per call), for every bitrate/codebook instance, fp16 and fp32 outputs.
Also checks that it is deterministic (bit-identical repeats).

    python -m pytest tests/test_turing_gemv_sk.py -v
"""

import os
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
    pytest.skip("sm_75 device required", allow_module_level=True)

DEV = "cuda:0"


def _make(k, n, K, m, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    trellis = torch.randint(0, 65536, (k // 16, n // 16, 16 * K), generator=g, dtype=torch.int32).to(torch.int16).to(DEV)
    suh = (torch.randn(k, generator=g) * 0.1).to(torch.float16).to(DEV)
    svh = (torch.randn(n, generator=g) * 0.1).to(torch.float16).to(DEV)
    A = torch.randn(m, k, generator=g).to(torch.float16).to(DEV)
    return A, trellis, suh, svh


def _gemm(A, trellis, suh, svh, mcg, mul1, dtype, sk):
    os.environ["EXL3_GEMV_SK"] = "1" if sk else "0"
    try:
        C = torch.empty((A.shape[0], trellis.shape[1] * 16), dtype=dtype, device=DEV)
        ext.exl3_gemm(A, trellis, C, suh, torch.empty_like(A), svh, -1, mcg, mul1, -1)
        return C
    finally:
        os.environ.pop("EXL3_GEMV_SK", None)


def _reference(A, trellis, suh, svh, K, mcg, mul1):
    k, n = trellis.shape[0] * 16, trellis.shape[1] * 16
    B = torch.empty((k, n), dtype=torch.float16, device=DEV)
    ext.reconstruct_had_slice(B, trellis, suh, svh, K, mcg, mul1, 0)
    return A.float() @ B.float()


CODEBOOKS = [(4, False, False), (4, True, False), (4, False, True), (3, True, False), (3, False, True), (2, True, False), (2, False, True)]


@pytest.mark.parametrize("K,mcg,mul1", CODEBOOKS)
@pytest.mark.parametrize("m", [1, 2, 5, 8])
@pytest.mark.parametrize("k,n", [(5120, 6144), (6144, 5120), (2048, 1024), (1024, 12288)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_gemv_sk_matches_reference(K, mcg, mul1, m, k, n, dtype):
    A, trellis, suh, svh = _make(k, n, K, m, seed = k * 31 + n + m * 7 + K)
    ref = _reference(A, trellis, suh, svh, K, mcg, mul1)
    y_sk = _gemm(A, trellis, suh, svh, mcg, mul1, dtype, True).float()
    y_old = _gemm(A, trellis, suh, svh, mcg, mul1, dtype, False).float()
    scale = ref.abs().max().item()
    err_sk = (y_sk - ref).abs().max().item() / scale
    err_old = (y_old - ref).abs().max().item() / scale
    _check(err_sk, err_old)


def _check(err_sk, err_old):
    # fp16 MMA accumulation (folded to fp32 every ring cycle) and fp16 weights: same error class as the old GEMV
    assert err_sk < 4e-3, f"split-k GEMV vs reference: {err_sk:.2e} (old GEMV {err_old:.2e})"
    assert err_sk < 2 * err_old + 1e-3, f"split-k GEMV error {err_sk:.2e} well above the old GEMV's {err_old:.2e}"


@pytest.mark.parametrize("K,mcg,mul1", [(5, False, True), (6, False, False), (6, True, False), (6, False, True), (8, True, False)])
@pytest.mark.parametrize("m", [1, 3, 8])
@pytest.mark.parametrize("k,n", [(5120, 4096), (1024, 131072)])
def test_gemv_sk_high_bitrate(K, mcg, mul1, m, k, n):
    # 5-8 bpw had no GEMV before (regular GEMM with EXL3_GEMV_SK=0); n = 131072 gives 2048 column groups, more than
    # there are warps, so ranges span whole groups that go straight to the workspace
    A, trellis, suh, svh = _make(k, n, K, m, seed = k + n + m * 7 + K)
    ref = _reference(A, trellis, suh, svh, K, mcg, mul1)
    y_sk = _gemm(A, trellis, suh, svh, mcg, mul1, torch.float16, True).float()
    y_old = _gemm(A, trellis, suh, svh, mcg, mul1, torch.float16, False).float()
    scale = ref.abs().max().item()
    _check((y_sk - ref).abs().max().item() / scale, (y_old - ref).abs().max().item() / scale)


@pytest.mark.parametrize("m", [1, 8])
def test_gemv_sk_deterministic(m):
    A, trellis, suh, svh = _make(17408 // 2, 5120, 4, m, seed = 5)
    ys = [_gemm(A, trellis, suh, svh, False, True, torch.float32, True) for _ in range(4)]
    for y in ys[1:]:
        assert torch.equal(y, ys[0])
