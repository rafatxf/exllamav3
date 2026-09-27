import ctypes
import math
import os
import sys

import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.attention_fn import fdq4

# fdq4 (flash-decoding from packed 4-bit caches on sm_75) against an fp32 reference on the dequantized cache:
# the eager kernel, and the graph-path cubins launched exactly as BC_Attention launches them (driver launch with
# the parameter lists of the Triton decode kernels, grid programs = bsz * n_kv_heads * h_blocks), with canaries
# around the output and partial buffers to catch out-of-bounds writes

device = "cuda:0"
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (7, 5),
    reason = "needs an sm_75+ CUDA device",
)
NQ, NKV, HD, PAGE = 24, 4, 256, 256
G = NQ // NKV
SCALE = 1 / math.sqrt(HD)


def _cache(n_pages, seed):
    gen = torch.Generator(device = device).manual_seed(seed)
    td = NKV * HD
    qk = torch.randint(-2**31, 2**31 - 1, (n_pages, PAGE, td // 8), dtype = torch.int32, device = device, generator = gen)
    qv = torch.randint(-2**31, 2**31 - 1, (n_pages, PAGE, td // 8), dtype = torch.int32, device = device, generator = gen)
    sk = (torch.rand((n_pages, PAGE, td // 32), device = device, generator = gen) * 0.5 + 0.05).half()
    sv = (torch.rand((n_pages, PAGE, td // 32), device = device, generator = gen) * 0.5 + 0.05).half()
    return qk, qv, sk, sv, gen


def _dequant(qk, qv, sk, sv):
    n_pages = qk.shape[0]
    kf = torch.empty((n_pages, PAGE, NKV, HD), dtype = torch.half, device = device)
    vf = torch.empty_like(kf)
    ext.dequant_cache_paged(
        qk, sk, kf, qv, sv, vf,
        torch.tensor([n_pages * PAGE], dtype = torch.int32, device = device),
        torch.arange(n_pages, dtype = torch.int32, device = device).unsqueeze(0), PAGE, -1, 0.0,
    )
    return kf, vf


def _reference(q, kf, vf, bt, seqlen, ql):
    L = seqlen + ql
    K = kf[bt[0].long()].reshape(-1, NKV, HD)[:L].float()
    V = vf[bt[0].long()].reshape(-1, NKV, HD)[:L].float()
    out = torch.empty((1, ql, NQ, HD), device = device)
    pos = torch.arange(L, device = device)[None, :]
    lim = (seqlen + torch.arange(ql, device = device))[:, None]
    for h in range(NQ):
        s = (q[0, :, h].float() @ K[:, h // G].T) * SCALE
        out[0, :, h] = torch.softmax(s.masked_fill(pos > lim, float("-inf")), -1) @ V[:, h // G]
    return out


def _rel(a, b):
    return ((a.float() - b).abs().max() / b.abs().max()).item()


@pytest.mark.parametrize("seqlen", [1, 100, 255, 256, 1000, 4133, 70001])
@pytest.mark.parametrize("ql", [1, 2, 4, 8])
def test_fdq4_eager(seqlen, ql):
    torch.cuda.set_device(device)
    n_pages = -(-(seqlen + ql) // PAGE) + 2
    qk, qv, sk, sv, gen = _cache(n_pages, seed = seqlen * 10 + ql)
    bt = torch.randperm(n_pages, device = device, generator = gen).to(torch.int32).unsqueeze(0).contiguous()
    q = (torch.randn((1, ql, NQ, HD), device = device, generator = gen) * 2).half()
    out = torch.empty_like(q)
    sl = torch.tensor([seqlen], dtype = torch.int32, device = device)
    ext.fdq4_decode(q, qk, sk, qv, sv, bt, sl, out, seqlen, ql, SCALE, 0, 1)
    kf, vf = _dequant(qk, qv, sk, sv)
    assert _rel(out, _reference(q, kf, vf, bt, seqlen, ql)) < 5e-3


def _driver():
    if os.name == "nt":
        return ctypes.WinDLL("nvcuda.dll")
    return ctypes.CDLL("libcuda.so.1")


def _get_fn(cu, image, name):
    mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
    assert cu.cuModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(image)) == 0
    assert cu.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode()) == 0
    return fn


def _launch(cu, fn, grid, threads, args):
    # TritonKernel-style launch: the listed params plus two trailing scratch pointers
    slots = [ctypes.c_uint64(a) for a in args] + [ctypes.c_uint64(0), ctypes.c_uint64(0)]
    ptrs = (ctypes.c_void_p * len(slots))(*[ctypes.cast(ctypes.pointer(s), ctypes.c_void_p) for s in slots])
    stream = torch.cuda.current_stream().cuda_stream
    assert cu.cuLaunchKernel(fn, grid[0], grid[1], 1, threads, 1, 1, 0, ctypes.c_void_p(stream), ptrs, None) == 0


def _h_blocks(ql):
    bm = 1
    while bm < ql:
        bm <<= 1
    return math.ceil(G / max(16 // bm, 1))


def _cdiv(a, b):
    return -(-a // b)


@pytest.mark.skipif(fdq4._find_nvcc() is None, reason = "graph-path cubins need nvcc")
@pytest.mark.parametrize("ql", [1, 2, 4, 8])
@pytest.mark.parametrize("seqlen", [5, 300, 4133, 70001])
def test_fdq4_graph_cubins(ql, seqlen):
    torch.cuda.set_device(device)
    cu = _driver()
    image = fdq4._bc_cubin(ql, NQ, NKV, SCALE)
    f_split, f_comb = _get_fn(cu, image, "fdq4_bc_split"), _get_fn(cu, image, "fdq4_bc_combine")
    programs, R = NKV * _h_blocks(ql), ql * G

    bt_width = _cdiv(seqlen + ql, PAGE) + 3                      # job table wider than the live length
    n_pages = bt_width + 2
    qk, qv, sk, sv, gen = _cache(n_pages, seed = seqlen + ql)
    bt = torch.randperm(n_pages, device = device, generator = gen)[:bt_width].to(torch.int32).unsqueeze(0).contiguous()
    sl = torch.tensor([seqlen], dtype = torch.int32, device = device)
    q = (torch.randn((1, ql, NQ, HD), device = device, generator = gen) * 2).half()

    splits_cap = max(1, 136 // NKV)
    bound = bt_width * PAGE + ql
    num_splits = max(1, min(splits_cap, _cdiv(bound, 16)))
    split_len = _cdiv(_cdiv(bound, num_splits), 16) * 16

    guard = torch.full((3, ql, NQ, HD), 7.0, dtype = torch.half, device = device)
    out = guard[1:2]
    po = torch.full((programs * splits_cap * R * HD + 4096,), 123.0, device = device)
    pml = torch.full((programs * splits_cap * R * 2 + 4096,), 123.0, device = device)
    _launch(cu, f_split, (programs, splits_cap), 128, [
        q.data_ptr(), qk.data_ptr(), qv.data_ptr(), bt.data_ptr(), sl.data_ptr(), out.data_ptr(), po.data_ptr(),
        pml.data_ptr(), sk.data_ptr(), sv.data_ptr(), 0, split_len, bt_width, num_splits, 0,
    ])
    _launch(cu, f_comb, (programs, R), 256, [po.data_ptr(), pml.data_ptr(), out.data_ptr(), 0, num_splits, 0])
    torch.cuda.synchronize()

    kf, vf = _dequant(qk, qv, sk, sv)
    assert _rel(out, _reference(q, kf, vf, bt, seqlen, ql)) < 3e-3
    assert (guard[0] == 7.0).all() and (guard[2] == 7.0).all(), "output canaries clobbered"
    assert (po[-4096:] == 123.0).all() and (pml[-4096:] == 123.0).all(), "partial buffers overrun"
