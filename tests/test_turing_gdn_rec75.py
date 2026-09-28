"""
sm_75 gated delta rule recurrent step kernel (cuda_recurrent_gated_delta_rule_kernel_128r in gdn.cu): state in
registers, one read and one write per step. Must match the generic kernel (EXL3_GDN_REC75=0, read per call) bit for
bit, in outputs and in every state it leaves (final state and per-step history), with and without slots.

    python -m pytest tests/test_turing_gdn_rec75.py -v
"""

import os
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5):
    pytest.skip("sm_75 device required", allow_module_level=True)

DEV = "cuda:0"


def _run(mixed_qkv, g, beta, state, nk, nv, slots, history, rec75):
    os.environ["EXL3_GDN_REC75"] = "1" if rec75 else "0"
    try:
        state = state.clone()
        bsz, seqlen = mixed_qkv.shape[:2]
        out = torch.empty((bsz, seqlen, nv, 128), dtype=torch.bfloat16, device=DEV)
        ext.cuda_recurrent_gated_delta_rule(mixed_qkv, g, beta, state, out, nk, nv, 128, 128, slots, history)
        return out, state
    finally:
        os.environ.pop("EXL3_GDN_REC75", None)


@pytest.mark.parametrize("nk,nv", [(16, 48), (16, 32)])
@pytest.mark.parametrize("seqlen", [1, 3, 8])
@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("use_slots", [False, True])
def test_rec75_matches_generic(nk, nv, seqlen, history, use_slots):
    torch.manual_seed(nk * 100 + nv + seqlen * 7 + history * 3 + use_slots)
    bsz = 1
    num_slots = 3 if use_slots else bsz
    max_history = 8
    mixed_qkv = torch.randn(bsz, seqlen, 2 * nk * 128 + nv * 128, device=DEV).to(torch.bfloat16)
    g = -torch.rand(bsz, seqlen, nv, device=DEV) * 2.0
    beta = torch.rand(bsz, seqlen, nv, device=DEV).to(torch.bfloat16)
    state = torch.randn(num_slots, max_history + 1, nv, 128, 128, device=DEV) * 0.1
    slots = torch.tensor([2], dtype=torch.int32, device=DEV) if use_slots else None
    out_a, st_a = _run(mixed_qkv, g, beta, state, nk, nv, slots, history, False)
    out_b, st_b = _run(mixed_qkv, g, beta, state, nk, nv, slots, history, True)
    assert torch.equal(out_a, out_b), (out_a.float() - out_b.float()).abs().max().item()
    assert torch.equal(st_a, st_b), (st_a - st_b).abs().max().item()
