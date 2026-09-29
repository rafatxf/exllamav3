"""
fdq4: flash-decoding straight from packed 4-bit K/V caches on Turing (sm_75), exllamav3_ext/turing/fdq4*.

On sm_75 the Triton paged decode kernels reach 17-50 GB/s at long context (tile ladders shrunk for 64 KB of
shared memory, 16-row q blocks for 6-head groups, K/V re-read per head block). fdq4 reads each packed K/V byte
once per (kv head, split) for all q rows of the GQA group and all draft positions at once: x8-11 at 128K.

Covers quant-direct decode / draft-verify calls with 4-bit K and V, head_dim 256, causal, no window, softcap or
sinks, and at most 48 q rows per kv head (q_len * group); anything else falls through to the Triton kernels.
EXL3_FDQ4 (default on for sm_75, see util/turing.py).

Two entry points: fn_fdq4_decode_qc for eager dispatch, and bc_eligible / bc_build for BC_Attention's CUDA-graph
slots. The graph path launches kernels with exactly the parameter lists of the AOT Triton decode kernels (so BC
can patch graph params by index), which bakes slot shape and softmax scale into the code: those cubins are
compiled per slot shape with nvcc at runtime and cached on disk. Without nvcc the graph path keeps Triton.
"""
import hashlib, math, os, shutil, subprocess
import torch
from .common import AttnArgs
from ...ext import exllamav3_ext as ext
from ...util.turing import turing_flag

_src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "exllamav3_ext", "turing")
_bc_failed = False


def fn_fdq4_decode_qc(args: AttnArgs) -> torch.Tensor | None:
    if args.q_cache is None or not turing_flag("FDQ4", args.q.device):
        return None
    qk, sk, qv, sv, k_bits, v_bits = args.q_cache
    if (
        k_bits != 4 or v_bits != 4 or args.dim != 256 or not args.causal or args.is_swa() or
        args.softcap or args.sinks is not None or args.non_causal_spans or args.cu_seqlens is not None or
        args.q.dtype != torch.float16 or qk.shape[1] != 256 or
        args.q_len * (args.num_q_heads // args.num_kv_heads) > 48
    ):
        return None
    q = args.q.contiguous()
    out = torch.empty_like(q)
    bt = args.block_table if args.block_table.dtype == torch.int32 else args.block_table.int()
    sl = args.cache_seqlens if args.cache_seqlens.dtype == torch.int32 else args.cache_seqlens.int()
    max_kv = args.max_kv_len if args.max_kv_len is not None else bt.shape[1] * 256
    scale = args.sm_scale if args.sm_scale is not None else 1.0 / math.sqrt(args.dim)
    ext.fdq4_decode(q, qk, sk, qv, sv, bt, sl, out, int(max_kv), args.q_len, float(scale), 0, 1)
    return out


# ---- BC_Attention (graph path) --------------------------------------------------------------------------------

def _no_window(window_size) -> bool:
    from .triton_paged import _normalize_window
    left, right = _normalize_window(window_size)
    return left < 0 and right < 0


def _window_left(window_size) -> int | None:
    """Left window of a causal sliding-window layer (-1: no window), None if fdq4 cannot express it"""
    from .triton_paged import _normalize_window
    left, right = _normalize_window(window_size)
    if right >= 0 and not (left >= 0 and right == 0):
        return None
    return left


def _h_blocks(q_len: int, group: int) -> int:
    # BC_Attention's program count per (batch, kv head), from the Triton q/head blocking (see fdq4_bc.cu.in)
    bm = 1
    while bm < q_len: bm <<= 1
    bh = max(16 // bm, 1)
    return (group + bh - 1) // bh


def _row_split(q_len: int, group: int) -> bool:
    # More than 48 q rows per kv head (e.g. Gemma 4 global layers, 8 q heads per kv head, at 8 verify positions):
    # split the rows over BC's per-kv-head programs, each re-reading that head's K/V
    return q_len * group > 48


def bc_eligible(module, q_len: int, causal: bool) -> bool:
    if _bc_failed or not turing_flag("FDQ4", module.device):
        return False
    # fp16 K/V (fd16 variant): SlidingAttention's window ring and fp16 caches
    quant = getattr(module, "quant", False)
    if not quant and not turing_flag("FD16", module.device):
        return False
    group = module.num_q_heads // module.num_kv_heads
    hd = module.head_dim
    rows = q_len * group
    if _row_split(q_len, group):
        hb = _h_blocks(q_len, group)
        rows = (rows + hb - 1) // hb
    return (
        (not quant or (module.k_bits == 4 and module.v_bits == 4)) and hd in (256, 512) and
        getattr(module, "v_head_dim", hd) == hd and causal and
        _window_left(module.window_size) is not None and not (module.softcap or 0.0) and module.sinks is None and
        rows <= 48
    )


def _find_nvcc() -> str | None:
    homes = [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")]
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        homes.append(CUDA_HOME)
    except Exception:
        pass
    exe = "nvcc.exe" if os.name == "nt" else "nvcc"
    for home in homes:
        if home and os.path.isfile(os.path.join(home, "bin", exe)):
            return os.path.join(home, "bin", exe)
    return shutil.which("nvcc")


def _bc_cubin(q_len: int, nq: int, nkv: int, scale: float, hd: int = 256, win_left: int = -1, fp16: bool = False) -> bytes:
    src = os.path.join(_src_dir, "fdq4_bc.cu.in")
    deps = [src, os.path.join(_src_dir, "fdq4_core.cuh")]
    defs = [f"-DFDQ4_QL={q_len}", f"-DFDQ4_NQ={nq}", f"-DFDQ4_NKV={nkv}", f"-DFDQ4_SCALE_LOG2={scale * 1.4426950408889634!r}f"]
    # Extra defines only when they differ from the head_dim 256 / no window defaults, so existing cubins keep
    # their cache keys
    if hd != 256: defs.append(f"-DFD_HD={hd}")
    if win_left >= 0: defs.append(f"-DFDQ4_WL={win_left}")
    if _row_split(q_len, nq // nkv): defs.append("-DFDQ4_ROWSPLIT=1")
    if fp16: defs.append("-DFD_FP16=1")
    h = hashlib.sha256(b"".join(open(f, "rb").read() for f in deps) + " ".join(defs).encode()).hexdigest()[:16]
    cache_dir = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "exllamav3", "fdq4")
    os.makedirs(cache_dir, exist_ok = True)
    out = os.path.join(cache_dir, f"fdq4_bc_{h}.cubin")
    if not os.path.exists(out):
        nvcc = _find_nvcc()
        if nvcc is None:
            raise RuntimeError("nvcc not found (set CUDA_HOME)")
        tmp = f"{out}.{os.getpid()}.tmp"
        subprocess.run(
            [nvcc, "-x", "cu", "-cubin", "-O3", "-arch=sm_75", "-lineinfo", "-I", _src_dir, *defs, "-o", tmp, src],
            check = True, capture_output = True,
        )
        os.replace(tmp, out)
    with open(out, "rb") as f:
        return f.read()


def bc_build(module, bsz: int, q_len: int):
    """
    -> (k_split, k_combine, programs, splits_cap, block_n, rows) for a BC slot, or None if the cubin cannot be
    built (then the graph path keeps the Triton kernels for the rest of the process)
    """
    global _bc_failed
    nq, nkv = module.num_q_heads, module.num_kv_heads
    hd = module.head_dim
    rows = q_len * (nq // nkv)
    try:
        cubin = _bc_cubin(q_len, nq, nkv, float(module.sm_scale), hd, _window_left(module.window_size),
                          not getattr(module, "quant", False))
    except Exception as e:
        detail = e.stderr.decode(errors = "replace")[-800:] if isinstance(e, subprocess.CalledProcessError) and e.stderr else str(e)
        print(f" !! fdq4: graph-path kernels unavailable, falling back to Triton decode attention: {detail}")
        _bc_failed = True
        return None
    k_split = ext.TritonKernel(cubin, "fdq4_bc_split", hd // 64, 0)
    k_combine = ext.TritonKernel(cubin, "fdq4_bc_combine", hd // 32, 0)
    k_combine.grid_y = rows
    programs = bsz * nkv
    sms = torch.cuda.get_device_properties(module.device).multi_processor_count
    rows_blk = rows
    if _row_split(q_len, nq // nkv):
        hb = _h_blocks(q_len, nq // nkv)
        rows_blk = (rows + hb - 1) // hb
        programs *= hb
    # Occupancy of the extension's head_dim 256 build (4 warps); 512-dim blocks are twice as wide
    per_sm = max(1, ext.fdq4_blocks_per_sm((rows_blk + 7) // 8) // (hd // 256))
    splits_cap = max(1, (per_sm * sms) // programs)
    return k_split, k_combine, programs, splits_cap, 16, rows
