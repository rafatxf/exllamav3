"""fa75 at head_dim 512 (Gemma 4 global layers) against fp32 attention: causal prefill chunks appended to a cache,
GQA 32 / 4, partial tiles, and the non-causal and windowed variants. Needs an sm_75 GPU."""
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5),
    reason = "sm_75 only",
)


def _ref(q, k, v, scale, causal, window):
    Tq, Hq, _ = q.shape
    Tkv, Hkv, _ = k.shape
    G = Hq // Hkv
    offs = Tkv - Tq
    i = torch.arange(Tq, device = q.device)[:, None]
    j = torch.arange(Tkv, device = q.device)[None, :]
    mask = (j <= i + offs) if causal else torch.ones(Tq, Tkv, dtype = torch.bool, device = q.device)
    if window >= 0:
        mask = mask & (j >= i + offs - window)
    out = torch.empty(q.shape, device = q.device)
    for h in range(Hq):
        s = (q[:, h].float() @ k[:, h // G].float().T) * scale
        out[:, h] = torch.softmax(s.masked_fill(~mask, float("-inf")), -1) @ v[:, h // G].float()
    return out


@pytest.mark.parametrize("Tq, Tkv, causal, window", [
    (64, 64, True, -1), (100, 100, True, -1), (2048, 2048, True, -1), (300, 1000, True, -1),
    (1000, 5000, True, -1), (17, 4000, True, -1), (130, 130, False, -1), (700, 2100, True, 1023),
])
def test_fa75_512(Tq, Tkv, causal, window):
    torch.manual_seed(0)
    q = (torch.randn(Tq, 32, 512, device = "cuda") * 0.05).half()
    k = torch.randn(Tkv, 4, 512, device = "cuda").half()
    v = torch.randn(Tkv, 4, 512, device = "cuda").half()
    o = torch.empty(Tq, 32, 512, device = "cuda", dtype = torch.half)
    ext.fa75_fwd_win(q, k, v, o, 1.0, causal, window)
    r = _ref(q, k, v, 1.0, causal, window)
    err = ((o.float() - r).abs().max() / r.abs().max()).item()
    assert err < 2e-3, err
