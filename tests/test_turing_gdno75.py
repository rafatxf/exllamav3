import os
import sys

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.vendor.fla.chunk_o import chunk_fwd_o

# gdno75 (gated delta rule output stage on mma.m16n8k8) against an fp32 reference and the batched-cuBLAS path, at
# Qwen3.8-27B GDN shapes (16 key heads, 48 value heads, K = V = 128)

device = "cuda:0"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (7, 5),
    reason = "needs an sm_75+ CUDA device",
)
BT = 64


def _reference(q, k, v, h, g, scale):
    B, T, H, _ = q.shape
    HV = v.shape[2]
    G = HV // H
    o = torch.zeros(v.shape, device = device)
    for b in range(B):
        for c in range((T + BT - 1) // BT):
            s, e = c * BT, min((c + 1) * BT, T)
            for hv in range(HV):
                qc, kc = q[b, s:e, hv // G].float(), k[b, s:e, hv // G].float()
                vc, hc, gc = v[b, s:e, hv].float(), h[b, c, hv].float(), g[b, s:e, hv].float()
                A = torch.tril((qc @ kc.T) * torch.exp2(gc[:, None] - gc[None, :]))
                o[b, s:e, hv] = ((qc @ hc) * torch.exp2(gc)[:, None] + A @ vc) * scale
    return o


@pytest.mark.parametrize("B, T", [(1, 1000), (1, 64), (1, 37), (2, 200)])
def test_gdno75_matches_reference(monkeypatch, B, T):
    torch.cuda.set_device(device)
    gen = torch.Generator(device = device).manual_seed(T + B)
    H, HV = 16, 48
    NT = (T + BT - 1) // BT
    q = torch.nn.functional.normalize(torch.randn(B, T, H, 128, device = device, generator = gen), dim = -1).half()
    k = torch.nn.functional.normalize(torch.randn(B, T, H, 128, device = device, generator = gen), dim = -1).half()
    v = torch.randn(B, T, HV, 128, device = device, dtype = torch.half, generator = gen)
    h = (torch.randn(B, NT, HV, 128, 128, device = device, generator = gen) * 0.3).half()
    graw = -torch.rand(B, T, HV, device = device, generator = gen) * 0.05
    g = torch.zeros_like(graw)
    for c in range(NT):
        g[:, c * BT:(c + 1) * BT] = graw[:, c * BT:(c + 1) * BT].cumsum(1)
    scale = 128 ** -0.5

    monkeypatch.setenv("EXL3_GDN_O_CUDA", "1")
    o = chunk_fwd_o(q = q, k = k, v = v, h = h, g = g, scale = scale, chunk_size = BT)
    torch.cuda.synchronize()
    ref = _reference(q, k, v, h, g, scale)
    assert ((o.float() - ref).abs().max() / ref.abs().max()).item() < 2e-3
