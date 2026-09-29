"""fd16 (flash-decoding over fp16 K/V) cubins, as the graph path compiles them, against fp32 attention over a
shuffled paged cache: head_dim 256 / 512, q_len 1 / 8, sliding window or none, lengths around window and chunk edges.
Needs an sm_75 GPU and nvcc."""
import pytest
import torch
from exllamav3.modules.attention_fn import fdq4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 5) or fdq4._find_nvcc() is None,
    reason = "sm_75 and nvcc only",
)

PAGE = 256
_kernels = {}


def _kernels_for(hd, nq, nkv, ql, win, scale):
    key = (hd, nq, nkv, ql, win, scale)
    if key not in _kernels:
        cubin = fdq4._bc_cubin(ql, nq, nkv, scale, hd, win, True)
        _kernels[key] = (fdq4._DrvKernel(cubin, "fdq4_bc_split", hd // 2), fdq4._DrvKernel(cubin, "fdq4_bc_combine", hd))
    return _kernels[key]


@pytest.mark.parametrize("hd, nq, nkv, ql, win", [
    (256, 16, 8, 1, 1023), (256, 16, 8, 8, 1023), (512, 16, 2, 1, -1), (512, 16, 2, 8, -1), (256, 24, 4, 1, -1),
])
@pytest.mark.parametrize("seqlen", [3, 40, 1022, 1023, 1024, 1025, 1040, 1063, 2047, 2800])
def test_fd16(hd, nq, nkv, ql, win, seqlen):
    torch.manual_seed(seqlen)
    scale = 1.0
    k_split, k_comb = _kernels_for(hd, nq, nkv, ql, win, scale)
    G = nq // nkv
    R = ql * G
    pps = 12
    K = torch.randn(pps, PAGE, nkv, hd, device = "cuda").half()
    V = torch.randn(pps, PAGE, nkv, hd, device = "cuda").half()
    bt = torch.randperm(pps, device = "cuda").int().view(1, pps).contiguous()
    q = (torch.randn(1, ql, nq, hd, device = "cuda") * 0.1).half()
    sl = torch.tensor([seqlen], dtype = torch.int32, device = "cuda")
    hb = fdq4._h_blocks(ql, G)
    programs = nkv * hb
    splits = 24
    split_len = -(-(-(-(pps * PAGE) // splits)) // 16) * 16
    po = torch.zeros(nkv * splits * R * hd, device = "cuda")
    pml = torch.zeros(nkv * splits * R * 2, device = "cuda")
    out = torch.zeros_like(q)
    dummy = torch.zeros(1, device = "cuda", dtype = torch.half)
    stream = torch.cuda.current_stream().cuda_stream
    k_split.launch((programs, splits, 1), [q, K, V, bt, sl, None, po, pml, dummy, dummy, None, split_len, pps, splits, None], stream)
    k_comb.launch((programs, R, 1), [po, pml, out, None, splits, None], stream)
    torch.cuda.synchronize()

    kv_len = seqlen + ql
    Kb = torch.cat([K[p] for p in bt[0].tolist()])[:kv_len].float()
    Vb = torch.cat([V[p] for p in bt[0].tolist()])[:kv_len].float()
    worst = 0.0
    for qi in range(ql):
        pos = seqlen + qi
        lo = max(0, pos - win) if win >= 0 else 0
        for h in range(nq):
            s = (Kb[lo:pos + 1, h // G] @ q[0, qi, h].float()) * scale
            ref = torch.softmax(s, 0) @ Vb[lo:pos + 1, h // G]
            worst = max(worst, ((out[0, qi, h].float() - ref).abs().max() / ref.abs().max()).item())
    assert worst < 3e-3, worst
