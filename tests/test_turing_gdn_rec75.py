"""
sm_75 gated delta rule recurrent step kernel (cuda_recurrent_gated_delta_rule_kernel_128r in gdn.cu): state in
registers, one read and one write per step, and lazy speculative history (initial state + step inputs, replayed on
rewind). Must match the generic kernel (EXL3_GDN_REC75=0, read per call by the extension) bit for bit: outputs,
final state, and for every accepted-step count the rewound state against the generic kernel's history slot.

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


def _inputs(nk, nv, bsz, seqlen, num_slots, max_history, seed):
    torch.manual_seed(seed)
    mixed_qkv = torch.randn(bsz, seqlen, 2 * nk * 128 + nv * 128, device=DEV).to(torch.bfloat16)
    g = -torch.rand(bsz, seqlen, nv, device=DEV) * 2.0
    beta = torch.rand(bsz, seqlen, nv, device=DEV).to(torch.bfloat16)
    state = torch.randn(num_slots, max_history + 1, nv, 128, 128, device=DEV) * 0.1
    return mixed_qkv, g, beta, state


@pytest.mark.parametrize("nk,nv", [(16, 48), (16, 32)])
@pytest.mark.parametrize("seqlen", [1, 3, 8])
@pytest.mark.parametrize("use_slots", [False, True])
def test_rec75_no_history_matches_generic(nk, nv, seqlen, use_slots):
    num_slots = 3 if use_slots else 1
    mixed_qkv, g, beta, state = _inputs(nk, nv, 1, seqlen, num_slots, 8, nk * 100 + nv + seqlen * 7 + use_slots)
    slots = torch.tensor([2], dtype=torch.int32, device=DEV) if use_slots else None
    out_a, st_a = _run(mixed_qkv, g, beta, state, nk, nv, slots, False, False)
    out_b, st_b = _run(mixed_qkv, g, beta, state, nk, nv, slots, False, True)
    assert torch.equal(out_a, out_b)
    assert torch.equal(st_a, st_b)


@pytest.mark.parametrize("nk,nv", [(16, 48), (16, 32)])
@pytest.mark.parametrize("seqlen", [1, 3, 8])
@pytest.mark.parametrize("max_history", [1, 8])
@pytest.mark.parametrize("bsz", [1, 2])
def test_rec75_history_and_rewind_match_generic(nk, nv, seqlen, max_history, bsz):
    if seqlen > max_history + 1:
        pytest.skip("history shorter than the verify")
    num_slots = 4
    mixed_qkv, g, beta, state = _inputs(nk, nv, bsz, seqlen, num_slots, max_history, nk + nv + seqlen + max_history * 3 + bsz)
    slots = torch.tensor([3, 1][:bsz], dtype=torch.int32, device=DEV)
    out_a, st_a = _run(mixed_qkv, g, beta, state, nk, nv, slots, True, False)
    out_b, st_b = _run(mixed_qkv, g, beta, state, nk, nv, slots, True, True)
    assert torch.equal(out_a, out_b)
    for s in slots.tolist():
        assert torch.equal(st_a[s, 0], st_b[s, 0]), "final state"
    lazy = max_history + 1 >= 3
    if not lazy:
        assert torch.equal(st_a, st_b)
        return
    # Rewind to every accepted-step count: replay from the lazy record vs the generic kernel's history slot
    for s in slots.tolist():
        for a in range(1, seqlen):
            st = st_b.clone()
            base = st.data_ptr() + s * st.stride(0) * st.element_size()
            ext.batched_state_rewind([ext.StateRewindJob(0, base, st.stride(1), a, nk, nv)], 0)
            assert torch.equal(st[s, 0], st_a[s, a]), f"slot {s}, {a} accepted steps"
