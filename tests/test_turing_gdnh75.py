import os
import sys

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.vendor.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from exllamav3.vendor.fla.gdn_chunk_fwd import chunk_gated_delta_rule_fwd_intra
from exllamav3.vendor.fla.cumsum import chunk_local_cumsum
from exllamav3.vendor.fla.l2norm import l2norm_fwd

# gdnh75 (Gated DeltaNet state recurrence on mma.m16n8k8) against the FLA Triton kernel it replaces, on inputs
# built by the real WY pipeline at Qwen3.8-27B GDN shapes (16 key heads, 48 value heads, K = V = 128)

device = "cuda:0"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (7, 5),
    reason = "needs an sm_75+ CUDA device",
)
BT = 64


def _inputs(B, T, H, HV, use_h0, seed):
    gen = torch.Generator(device = device).manual_seed(seed)
    k = l2norm_fwd(torch.randn(B, T, H, 128, device = device, dtype = torch.half, generator = gen))[0]
    v = torch.randn(B, T, HV, 128, device = device, dtype = torch.half, generator = gen)
    graw = -torch.rand(B, T, HV, device = device, generator = gen) * 0.1
    beta = torch.rand(B, T, HV, device = device, generator = gen).half()
    g = chunk_local_cumsum(graw, chunk_size = BT, scale = 1.4426950216)
    w, u, _ = chunk_gated_delta_rule_fwd_intra(k = k, v = v, g = g, beta = beta, chunk_size = BT)
    h0 = torch.randn(B, HV, 128, 128, device = device, generator = gen) * 0.5 if use_h0 else None
    return k, w, u, g, h0


def _rel(a, b):
    # relative to the reference magnitude; absolute when the reference is exactly zero (a single chunk without
    # an initial state stores h = 0)
    den = b.float().abs().max().item()
    return (a.float() - b.float()).abs().max().item() / (den if den > 0 else 1.0)


@pytest.mark.parametrize("B, T", [(1, 2048), (1, 1000), (1, 37), (2, 300)])
@pytest.mark.parametrize("use_h0", [True, False])
def test_gdnh75_matches_triton(monkeypatch, B, T, use_h0):
    torch.cuda.set_device(device)
    H, HV = 16, 48
    k, w, u, g, h0 = _inputs(B, T, H, HV, use_h0, seed = T)

    monkeypatch.setenv("EXL3_GDN_H_CUDA", "0")
    h_ref, vn_ref, ht_ref = chunk_gated_delta_rule_fwd_h(
        k = k, w = w, u = u, g = g, initial_state = h0, output_final_state = True, chunk_size = BT,
    )

    NT = (T + BT - 1) // BT
    h = torch.empty(B, NT, HV, 128, 128, device = device, dtype = torch.half)
    vn = torch.empty_like(u)
    ht = torch.empty(B, HV, 128, 128, device = device)
    ext.gdnh75_fwd(k, w, u, g, h0, h, vn, ht)
    torch.cuda.synchronize()

    # fp16 storage of h / v_new bounds the agreement (measured ~2-4e-4)
    assert _rel(h, h_ref) < 2e-3
    assert _rel(vn, vn_ref) < 2e-3
    assert _rel(ht, ht_ref) < 2e-3


def test_gdnh75_dispatch(monkeypatch):
    # chunk_gated_delta_rule_fwd_h takes the gdnh75 path when EXL3_GDN_H_CUDA is on, with the same result
    torch.cuda.set_device(device)
    k, w, u, g, h0 = _inputs(1, 513, 16, 48, True, seed = 1)
    monkeypatch.setenv("EXL3_GDN_H_CUDA", "0")
    ref = chunk_gated_delta_rule_fwd_h(k = k, w = w, u = u, g = g, initial_state = h0, output_final_state = True, chunk_size = BT)
    monkeypatch.setenv("EXL3_GDN_H_CUDA", "1")
    out = chunk_gated_delta_rule_fwd_h(k = k, w = w, u = u, g = g, initial_state = h0, output_final_state = True, chunk_size = BT)
    for a, b in zip(out, ref):
        assert _rel(a, b) < 2e-3
