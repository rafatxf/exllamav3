"""fa75 with a causal sliding window (fa75_fwd_win) against fp32 attention. Needs an sm_75 GPU."""
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5),
    reason = "sm_75 only",
)


def _ref(q, k, v, scale, window):
    Tq, Hq, _ = q.shape
    Tkv, Hkv, _ = k.shape
    G = Hq // Hkv
    offs = Tkv - Tq
    i = torch.arange(Tq, device = q.device)[:, None]
    j = torch.arange(Tkv, device = q.device)[None, :]
    mask = j <= i + offs
    if window >= 0:
        mask = mask & (j >= i + offs - window)
    out = torch.empty(q.shape, device = q.device)
    for h in range(Hq):
        s = (q[:, h].float() @ k[:, h // G].float().T) * scale
        out[:, h] = torch.softmax(s.masked_fill(~mask, float("-inf")), -1) @ v[:, h // G].float()
    return out


@pytest.mark.parametrize("Tq, Tkv, window", [
    (2048, 3100, 1023), (2048, 2048, 1023), (300, 900, 100), (517, 1777, 1023),
    (64, 1500, 1023), (2048, 3327, 1023), (100, 100, 5), (2048, 2600, -1),
])
def test_fa75_window(Tq, Tkv, window):
    torch.manual_seed(0)
    q = (torch.randn(Tq, 16, 256, device = "cuda") * 0.5).half()
    k = torch.randn(Tkv, 8, 256, device = "cuda").half()
    v = torch.randn(Tkv, 8, 256, device = "cuda").half()
    o = torch.empty(Tq, 16, 256, device = "cuda", dtype = torch.half)
    ext.fa75_fwd_win(q, k, v, o, 0.0625, True, window)
    r = _ref(q, k, v, 0.0625, window)
    err = ((o.float() - r).abs().max() / r.abs().max()).item()
    assert err < 2e-3, err
