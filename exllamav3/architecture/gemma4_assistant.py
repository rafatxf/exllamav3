from __future__ import annotations
from typing_extensions import override
import os
import weakref
import torch

from ..model.config import Config, no_default
from ..model.model import Model
from ..modules import Module, Linear, RMSNorm, TransformerBlock, GatedMLP, Embedding, Attention, SlidingAttention
from ..modules.attention_fn.common import AttnArgs
from ..modules.attention_fn import fdq4
from ..util.rope import RopeStyle, RoPE
from ..util.tensor import get_for_device, to2
from ..util.device_copy import to_device
from ..constants import PAGE_SIZE

# Gemma 4 MTP drafter ("assistant", Gemma4AssistantForCausalLM, e.g. google/gemma-4-31B-it-assistant).
#
# A small Gemma 4 decoder (4 layers, hidden 1024) with no K/V of its own: every layer is a KV-shared layer whose
# queries attend to the target's K/V, sliding layers to the target's last sliding-window layer and full layers to its
# last full-attention layer. Each draft step takes [target embedding of the last token || carried state] (2 x the
# target's hidden size) through pre_projection, runs the layers at one constant position (the position of the last
# token, i.e. the target's cache length), and returns the argmax of its own head (tied to its 1024-wide embedding
# table) plus post_projection of the normed state, which is carried into the next step. The first step carries the
# target's post-final-norm hidden state of the position that produced the token, as for Qwen3.5 MTP.
#
# The drafted tokens never enter the target's cache, so every step of a round sees the same keys: the target's
# positions 0 .. L-1 (L = cache length), the last sliding_window of them for sliding layers. Reference:
# transformers SinglePositionMultiTokenCandidateGenerator and Gemma4AssistantForCausalLM.


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _emb_gather_kernel(table_ptr, ids_ptr, out_ptr, D: tl.constexpr, scale, BLOCK: tl.constexpr):
        r = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = offs < D
        tok = tl.load(ids_ptr + r).to(tl.int64)
        v = tl.load(table_ptr + tok * D + offs, mask = m).to(tl.float32)
        # Same rounding as Embedding.forward: fp16 row times the multiplier, computed in fp32
        tl.store(out_ptr + r * D + offs, (v * scale).to(tl.float16), mask = m)
except ImportError:
    triton = None


class _MappedTable:
    """The target's CPU embedding table registered as mapped pinned memory, so kernels gather rows from it over the
    bus without a host round trip (with unified addressing the host pointer is the device pointer)"""

    def __init__(self, t: torch.Tensor):
        import ctypes
        self.cu = ctypes.CDLL("libcuda.so.1" if os.name != "nt" else "nvcuda.dll")
        self.t = t
        self.dtype = t.dtype
        r = self.cu.cuMemHostRegister_v2(ctypes.c_void_p(t.data_ptr()), ctypes.c_size_t(t.numel() * t.element_size()),
                                         ctypes.c_uint(2))     # CU_MEMHOSTREGISTER_DEVICEMAP
        # 712: already page-locked (e.g. a pinned tensor), which maps it as well on unified-addressing systems
        assert r in (0, 712), f"cuMemHostRegister failed ({r})"
        self.registered = r == 0
        dptr = ctypes.c_uint64()
        r = self.cu.cuMemHostGetDevicePointer_v2(ctypes.byref(dptr), ctypes.c_void_p(t.data_ptr()), ctypes.c_uint(0))
        if r != 0:
            self.release()
            raise RuntimeError(f"cuMemHostGetDevicePointer failed ({r})")
        self.ptr = dptr.value

    def data_ptr(self):
        return self.ptr

    def release(self):
        import ctypes
        if self.t is not None and self.registered:
            self.cu.cuMemHostUnregister(ctypes.c_void_p(self.t.data_ptr()))
        self.t = None


class _RoundGraph:
    pass


class Gemma4AssistantConfig(Config):
    arch_string = "Gemma4AssistantForCausalLM"

    def __init__(
        self,
        directory: str,
        **kwargs,
    ):
        super().__init__(
            directory,
            {"text": Gemma4AssistantModel},
            **kwargs
        )

        self.backbone_hidden_size = self.read_cfg(int, "backbone_hidden_size", no_default)
        if self.read_cfg(bool, "use_ordered_embeddings", False):
            raise NotImplementedError("Gemma 4 assistant: ordered (centroid-masked) embeddings are not implemented")

        self.num_hidden_layers = self.read_cfg(int, "text_config->num_hidden_layers", no_default)
        self.hidden_size = self.read_cfg(int, "text_config->hidden_size", no_default)
        self.head_dim = self.read_cfg(int, "text_config->head_dim", no_default)
        self.global_head_dim = self.read_cfg(int, "text_config->global_head_dim", self.head_dim)
        self.num_q_heads = self.read_cfg(int, "text_config->num_attention_heads", no_default)
        self.layer_types = self.read_cfg(list, "text_config->layer_types", no_default)
        assert len(self.layer_types) == self.num_hidden_layers
        self.sliding_window = self.read_cfg(int, "text_config->sliding_window", no_default)
        self.assert_cfg(str, "text_config->hidden_activation", "gelu_pytorch_tanh", True)
        self.intermediate_size = self.read_cfg(int, "text_config->intermediate_size", no_default)
        self.rms_norm_eps = self.read_cfg(float, "text_config->rms_norm_eps", no_default)
        self.vocab_size = self.read_cfg(int, "text_config->vocab_size", no_default)
        self.tie_word_embeddings = self.read_cfg(bool, "text_config->tie_word_embeddings", True)
        self.num_kv_shared_layers = self.read_cfg(int, "text_config->num_kv_shared_layers", self.num_hidden_layers)
        assert self.num_kv_shared_layers == self.num_hidden_layers, \
            "Gemma 4 assistant: only drafters whose every layer shares the target's K/V are supported"

        self.rope_settings_local = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 10000.0,
            config_dict = self.read_cfg(dict, "text_config->rope_parameters->sliding_attention", {}),
            override_type = self.read_cfg(str, "text_config->rope_parameters->sliding_attention->rope_type", None),
        )
        self.rope_settings_global = self.read_rope_settings_default(
            RopeStyle.NEOX,
            default_rope_theta = 1000000.0,
            config_dict = self.read_cfg(dict, "text_config->rope_parameters->full_attention", {}),
            override_type = self.read_cfg(str, "text_config->rope_parameters->full_attention->rope_type", None),
            override_head_dim = self.global_head_dim,
        )

        self.vision = None


class Gemma4AssistantInputLayer(Module):
    """[target embedding of the token || carried state] -> pre_projection"""

    def __init__(self, config: Config, key: str, backbone_hidden_size: int, hidden_size: int):
        super().__init__(config, key, None)
        self.module_name = "Gemma4AssistantInputLayer"
        self.out_dtype = torch.float
        self.proj = Linear(
            config = config,
            key = key,
            in_features = 2 * backbone_hidden_size,
            out_features = hidden_size,
            qmap = "g4a.in",
            out_dtype = torch.float,
            pad_to = 1,
        )
        self.register_submodule(self.proj)
        self.attached_model = None
        self.caps.update({"x_cpu": True})

    def optimizer_targets(self):
        raise NotImplementedError()

    def prepare_for_device(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None):
        h = get_for_device(params, "target_hidden", self.device)
        assert h.shape[:-1] == x.shape, \
            f"Gemma 4 assistant token/state shape mismatch: {tuple(x.shape)} vs {tuple(h.shape)}"
        e = self.attached_model().modules[0].forward(x, params, out_dtype = torch.half)
        y = torch.cat((to_device(e, self.device).half(), h.half()), dim = -1)
        y = self.proj.forward(y, params)
        return to2(y, out_dtype, self.out_dtype)


class Gemma4AssistantAttention(Module):
    """Query-only attention over the target's K/V (sliding: the target's fp16 state ring of its last sliding
    layer; full: the target's paged cache of its last full-attention layer, through the fdq4 draft-block kernels
    when available)"""

    def __init__(
        self,
        config: Config,
        key: str,
        layer_idx: int,
        hidden_size: int,
        head_dim: int,
        num_q_heads: int,
        rope_settings,
        sliding_window: int,
        rms_norm_eps: float,
    ):
        super().__init__(config, key, None)
        self.module_name = "Gemma4AssistantAttention"
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.rope_settings = rope_settings
        self.sliding_window = sliding_window     # > 0: attend to the last sliding_window target positions
        self.out_dtype = torch.float
        self.rope = None
        self._fd16_kernels = None
        self._dummy = None
        self.target_attn = None                  # weakref to the target layer whose K/V this layer reads

        self.q_proj = Linear(config, f"{key}.q_proj", hidden_size, num_q_heads * head_dim, qmap = f"g4a.{layer_idx}.q",
                             out_dtype = torch.half)
        self.o_proj = Linear(config, f"{key}.o_proj", num_q_heads * head_dim, hidden_size, qmap = f"g4a.{layer_idx}.o",
                             out_dtype = torch.float)
        self.q_norm = RMSNorm(config, f"{key}.q_norm", rms_norm_eps = rms_norm_eps, out_dtype = torch.half)
        self.register_submodule(self.q_proj)
        self.register_submodule(self.o_proj)
        self.register_submodule(self.q_norm)

    def optimizer_targets(self):
        raise NotImplementedError()

    @override
    def load(self, device: torch.device, **kwargs):
        super().load(device, **kwargs)
        self.rope = RoPE(device, self.rope_settings)
        self._dummy = torch.zeros(1, dtype = torch.half, device = device)

    @override
    def unload(self):
        self.rope = None
        self._dummy = None
        self._fd16_kernels = None
        super().unload()

    def _attend_sliding(self, q, params):
        # q: (bsz, 1, nq, hd). The target's state ring holds positions window_beg .. position - 1 of each sequence at
        # ring rows 0 .. position - window_beg - 1
        tattn = self.target_attn()
        rsg = params["target_recurrent_states"]
        rsl = rsg[0].cache.get_recurrent_layer((tattn.layer_idx, 0))
        k_states, v_states = rsl.get_state_tensors()
        nkv = tattn.num_kv_heads
        bsz, _, nq, hd = q.shape
        g = nq // nkv
        out = torch.empty_like(q)
        for b, rs in enumerate(rsg):
            hi = rs.position - rs.window_beg
            lo = max(0, hi - self.sliding_window)
            k = k_states[rs.slot, lo:hi]                                      # (n, nkv, hd)
            v = v_states[rs.slot, lo:hi]
            qb = q[b, 0].view(nkv, g, hd)
            s = torch.bmm(qb, k.permute(1, 2, 0)).float()                     # (nkv, g, n), scale 1
            p = torch.softmax(s, dim = -1).half()
            o = torch.bmm(p, v.permute(1, 0, 2))                              # (nkv, g, hd)
            out[b, 0] = o.view(nq, hd)
        return out

    def _attend_full(self, q, params):
        tattn = self.target_attn()
        cache = params["target_cache"]
        layer = cache.layers[tattn.layer_idx, 0]
        block_table = get_for_device(params, "block_table", self.device)
        seqlens = get_for_device(params, "g4a_base_seqlens", self.device)
        # Query at the target's cache length L attending to positions 0 .. L-1: a causal one-row call whose own
        # position is L - 1 covers exactly those keys, and nothing is written
        prev = (seqlens - 1).to(torch.int32)
        from ..cache import CacheLayer_quant
        if isinstance(layer, CacheLayer_quant):
            if params.get("g4a_graph"):
                prev = params["g4a_prev_seqlens"]
            args = AttnArgs(
                q.shape[0], 1, self.num_q_heads, self.head_dim,
                0, tattn.num_kv_heads,
                q, None, None,
                None, None,
                True,
                1.0,
                None, None,
                None,
                0.0,
                block_table, prev,
                q_cache = layer.get_qkv(),
                # The graph serves every round at this block-table width, so its split length can't depend on the
                # round's context length
                max_kv_len = None if params.get("g4a_graph") else int(params["g4a_base_seqlens_max"]),
            )
            o = fdq4.fn_fdq4_block_qc(args)
            if o is not None:
                return o
        # Fallback: dequantize / gather each row's keys and attend in PyTorch
        k_all, v_all = layer.get_kv(seqlens, block_table, -1)
        out = torch.empty_like(q)
        nkv = tattn.num_kv_heads
        g = self.num_q_heads // nkv
        for b in range(q.shape[0]):
            L = int(params["g4a_base_seqlens_host"][b])
            pages = block_table[b, : -(-L // PAGE_SIZE)].long()
            k = k_all[pages].flatten(0, 1)[:L]
            v = v_all[pages].flatten(0, 1)[:L]
            qb = q[b, 0].view(nkv, g, self.head_dim)
            s = torch.bmm(qb.float(), k.permute(1, 2, 0).float())
            p = torch.softmax(s, dim = -1)
            out[b, 0] = torch.bmm(p, v.permute(1, 0, 2).float()).half().view(self.num_q_heads, self.head_dim)
        return out

    def _attend_sliding_fd16(self, q, params):
        # Graph path: the fd16 flash-decoding kernels over the target's state ring viewed as pages (row b's pages at
        # slot * pps + j), one causal row at ring index hi - 1 with the window reaching back sliding_window keys
        tattn = self.target_attn()
        rsl = params["target_recurrent_states"][0].cache.get_recurrent_layer((tattn.layer_idx, 0))
        k_states, v_states = rsl.get_state_tensors()
        nkv = tattn.num_kv_heads
        bsz, _, nq, hd = q.shape
        kern = self._fd16_kernels
        if kern is None:
            cubin = fdq4._bc_cubin(1, nq, nkv, 1.0, hd, self.sliding_window - 1, True)
            kern = self._fd16_kernels = (fdq4._DrvKernel(cubin, "fdq4_bc_split", hd // 2),
                                         fdq4._DrvKernel(cubin, "fdq4_bc_combine", hd))
        k_split, k_comb = kern
        g = nq // nkv
        hb = fdq4._h_blocks(1, g)
        programs = bsz * nkv * hb
        pps = k_states.shape[1] // PAGE_SIZE
        sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        splits = max(1, min(pps * PAGE_SIZE // 64, (2 * sms) // programs))
        split_len = -(-(-(-(pps * PAGE_SIZE) // splits)) // 16) * 16
        po = torch.empty(bsz * nkv * splits * g * hd, dtype = torch.float, device = q.device)
        pml = torch.empty(bsz * nkv * splits * g * 2, dtype = torch.float, device = q.device)
        out = torch.empty_like(q)
        stream = torch.cuda.current_stream(q.device).cuda_stream
        k_split.launch((programs, splits, 1),
                       [q, k_states, v_states, params["g4a_swa_bt"], params["g4a_swa_sl"], None, po, pml,
                        self._dummy, self._dummy, None, split_len, pps, splits, None], stream)
        k_comb.launch((programs, g, 1), [po, pml, out, None, splits, None], stream)
        return out

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        assert seqlen == 1
        q = self.q_proj.forward(x, params).view(bsz, 1, self.num_q_heads, self.head_dim)
        q = self.q_norm.forward(q, params, out_dtype = torch.half)
        q, _ = self.rope.apply(q, None, 0, get_for_device(params, "g4a_base_seqlens", self.device), None, False)
        if params.get("g4a_graph"):
            o = self._attend_sliding_fd16(q, params) if self.sliding_window > 0 else self._attend_full(q, params)
        else:
            o = self._attend_sliding(q, params) if self.sliding_window > 0 else self._attend_full(q, params)
        o = self.o_proj.forward(o.reshape(bsz, 1, self.num_q_heads * self.head_dim), params)
        return to2(o, out_dtype, self.out_dtype)


class Gemma4AssistantOutputLayer(Module):
    """Final norm, then post_projection of the normed state (carried into the next step, the module output) and
    the drafter's own head on it (sample_from_state)"""

    def __init__(self, config: Config, key_norm: str, key_post: str, hidden_size: int, backbone_hidden_size: int,
                 vocab_size: int, rms_norm_eps: float):
        super().__init__(config, key_post, None)
        self.module_name = "Gemma4AssistantOutputLayer"
        self.out_dtype = torch.half
        self.norm = RMSNorm(config, key_norm, rms_norm_eps = rms_norm_eps, out_dtype = torch.half)
        self.post = Linear(config, key_post, hidden_size, backbone_hidden_size, qmap = "g4a.out",
                           out_dtype = torch.half, pad_to = 1)
        self.head = Linear(config, "lm_head", hidden_size, vocab_size, qmap = "g4a.out",
                           alt_key = "model.embed_tokens", out_dtype = torch.half, pad_to = 1,
                           caps = {"logits_output": True})
        self.register_submodule(self.norm)
        self.register_submodule(self.post)
        self.register_submodule(self.head)

    def optimizer_targets(self):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor, params: dict, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        h = self.norm.forward(x, params, out_dtype = torch.half)
        params["g4a_head_in"] = h
        return self.post.forward(h, params)


class Gemma4AssistantModel(Model):
    config_class = Gemma4AssistantConfig

    def __init__(self, config: Gemma4AssistantConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.input_layer = Gemma4AssistantInputLayer(
            config, "pre_projection", config.backbone_hidden_size, config.hidden_size
        )
        self.modules += [self.input_layer]
        self.first_block_idx = len(self.modules)
        self.attn_modules = []

        for idx in range(config.num_hidden_layers):
            is_full = config.layer_types[idx] == "full_attention"
            attn = Gemma4AssistantAttention(
                config,
                f"model.layers.{idx}.self_attn",
                layer_idx = idx,
                hidden_size = config.hidden_size,
                head_dim = config.global_head_dim if is_full else config.head_dim,
                num_q_heads = config.num_q_heads,
                rope_settings = config.rope_settings_global if is_full else config.rope_settings_local,
                sliding_window = -1 if is_full else config.sliding_window,
                rms_norm_eps = config.rms_norm_eps,
            )
            self.attn_modules.append(attn)
            mlp = GatedMLP(
                config = config,
                key = f"model.layers.{idx}.mlp",
                hidden_size = config.hidden_size,
                intermediate_size = config.intermediate_size,
                key_up = "up_proj",
                key_gate = "gate_proj",
                key_down = "down_proj",
                qmap = f"g4a.{idx}.mlp",
                activation_fn = "gelu",
                interm_dtype = torch.half,
                out_dtype = torch.float,
            )
            self.modules.append(TransformerBlock(
                config = config,
                key = f"model.layers.{idx}",
                layer_idx = idx,
                key_layer_scalar = "layer_scalar",
                attn_norm = RMSNorm(config, f"model.layers.{idx}.input_layernorm", rms_norm_eps = config.rms_norm_eps),
                attn = attn,
                attn_post_norm = RMSNorm(config, f"model.layers.{idx}.post_attention_layernorm",
                                         rms_norm_eps = config.rms_norm_eps, out_dtype = torch.float),
                mlp_norm = RMSNorm(config, f"model.layers.{idx}.pre_feedforward_layernorm",
                                   rms_norm_eps = config.rms_norm_eps),
                mlp = mlp,
                mlp_post_norm = RMSNorm(config, f"model.layers.{idx}.post_feedforward_layernorm",
                                        rms_norm_eps = config.rms_norm_eps, out_dtype = torch.float),
            ))

        self.last_kv_module_idx = len(self.modules) - 1
        self.output_layer = Gemma4AssistantOutputLayer(
            config, "model.norm", "post_projection", config.hidden_size, config.backbone_hidden_size,
            config.vocab_size, config.rms_norm_eps,
        )
        self.modules.append(self.output_layer)
        self.logit_layer_idx = len(self.modules) - 1

        self.caps.update({
            "supports_tp": False,
            "attach_target": True,
            "mtp_draft": True,
            "mtp_shared_kv": True,          # reads the target's K/V, no draft cache of its own
            "default_draft_size": 6,        # the checkpoint's num_assistant_tokens
            "autosplit_load_fwd": False,
        })
        self.attached_model = None
        self._table = None
        self._graphs = {}

    @override
    def unload(self):
        self._graphs = {}
        if isinstance(self._table, _MappedTable):
            self._table.release()
        self._table = None
        super().unload()

    @override
    def prepare_inputs(self, input_ids: torch.Tensor, params: dict) -> torch.Tensor:
        # Every step of a round runs at the position of the round's first input token (the target's cache length),
        # which the generator's cache_seqlens holds on the first step and advances afterwards
        if params.get("draft_step", 0) == 0 or "g4a_base_seqlens" not in params:
            sl = params["cache_seqlens"]
            self._round_seqlens = sl.to(torch.int32).clone()
            self._round_seqlens_host = sl.tolist() if sl.device.type == "cpu" else sl.cpu().tolist()
        params["g4a_base_seqlens"] = self._round_seqlens
        params["g4a_base_seqlens_host"] = self._round_seqlens_host
        params["g4a_base_seqlens_max"] = max(self._round_seqlens_host)
        return input_ids

    @override
    def default_chat_prompt(self, prompt: str, system_prompt: str = None) -> str:
        raise NotImplementedError("Gemma 4 assistant draft model does not have its own chat template")

    @override
    def prefill(self, input_ids: torch.Tensor, params: dict | None = None):
        # No draft cache: accepted tokens need no catching up
        return None

    def attach_to(self, target):
        self.attached_model = weakref.ref(target)
        self.input_layer.attached_model = weakref.ref(target)
        assert isinstance(target.modules[0], Embedding), "Expected the target's Embedding as its first module"
        assert target.config.hidden_size == self.config.backbone_hidden_size, \
            "Gemma 4 assistant: backbone_hidden_size doesn't match the target's hidden size"

        # The target's last sliding-window and last full-attention layers
        last = {}
        for m in target.modules:
            a = getattr(m, "attn", None)
            if isinstance(a, SlidingAttention):
                last["sliding"] = a
            elif isinstance(a, Attention):
                last["full"] = a
        for attn in self.attn_modules:
            t = last["sliding" if attn.sliding_window > 0 else "full"]
            assert t.head_dim == attn.head_dim, "Gemma 4 assistant: head_dim doesn't match the target's layer"
            attn.target_attn = weakref.ref(t)

        target_norm = target.modules[target.logit_layer_idx - 1]
        assert isinstance(target_norm, RMSNorm), "Expected target final RMSNorm immediately before lm_head"
        self.draft_verifier_params.update({
            "export_state_norm_keys": {target_norm.key},
        })

    def default_load_shape_dtype(self, chunk_size):
        return (1, 1), torch.long

    def default_load_params(self, max_chunk_size):
        return {}

    def sample_from_state(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        h = params.pop("g4a_head_in")
        logits = self.output_layer.head.forward(h, params)
        return torch.argmax(logits, dim = -1)

    # Whole drafting round as one CUDA graph (sm_75 fdq4 / fd16 kernels, Triton gather from the mapped embedding
    # table): no host round trip per step, one sync per round. None when unavailable; the generator then runs its
    # per-step MTP loop

    _round_failed = False
    _max_graphs = 8     # one per (batch size, window, block-table width): the width grows with the context

    def _graph_ok(self, params) -> bool:
        from ..util.turing import turing_flag
        dev = self.output_layer.device
        return (
            not self._round_failed and triton is not None and dev is not None and torch.device(dev).type == "cuda" and
            turing_flag("FDQ4", dev) != 0 and turing_flag("FDQ4_BLOCK", dev) != 0 and fdq4._find_nvcc() is not None and
            params.get("target_recurrent_states") is not None and params.get("target_cache") is not None and
            os.environ.get("EXL3_G4A_GRAPH", "1") != "0"
        )

    def _embed_rows(self, ids: torch.Tensor, out: torch.Tensor):
        emb = self.attached_model().modules[0]
        if self._table is None:
            w = emb.embedding.weight.data
            self._table = _MappedTable(w) if w.device.type == "cpu" else w
        D = out.shape[-1]
        _emb_gather_kernel[(ids.numel(), triton.cdiv(D, 1024))](self._table, ids, out, D, float(emb.multiplier), BLOCK = 1024)

    def _round_body(self, st):
        cur = st.ids0
        h = st.h0
        params = st.params
        for step in range(st.window):
            e = st.emb
            self._embed_rows(cur, e)
            x = torch.cat((e, h), dim = -1)
            x = self.input_layer.proj.forward(x, params)
            for m in self.modules[self.first_block_idx : self.last_kv_module_idx + 1]:
                x = m.forward(x, params)
            h = self.output_layer.forward(x, params)
            logits = self.output_layer.head.forward(params.pop("g4a_head_in"), params)
            cur = torch.argmax(logits, dim = -1)
            st.ids_out[:, step : step + 1].copy_(cur)

    def draft_round(self, ids: torch.Tensor, hidden: torch.Tensor, params: dict, window: int) -> torch.Tensor | None:
        """ids (bsz, 1) pinned host, hidden (bsz, 1, backbone) on device; params: block_table and cache_seqlens
        (pinned host), target_cache, target_recurrent_states. Returns (bsz, window) drafted ids on the device"""
        if not self._graph_ok(params):
            return None
        dev = self.output_layer.device
        rsg = params["target_recurrent_states"]
        bt = params["block_table"]
        bsz = ids.shape[0]
        rsl = rsg[0].cache.get_recurrent_layer((self.attn_modules[0].target_attn().layer_idx, 0))
        pps = rsl.get_state_tensors()[0].shape[1] // PAGE_SIZE
        key = (bsz, window, bt.shape[1], pps, id(params["target_cache"]))
        st = self._graphs.pop(key, None)
        fresh = st is None
        if not fresh:
            self._graphs[key] = st              # most recently used last
        else:
            while len(self._graphs) >= self._max_graphs:
                self._graphs.pop(next(iter(self._graphs)))
            st = _RoundGraph()
            st.window = window
            st.ids0 = torch.zeros((bsz, 1), dtype = torch.long, device = dev)
            st.h0 = torch.zeros((bsz, 1, self.config.backbone_hidden_size), dtype = torch.half, device = dev)
            st.emb = torch.zeros((bsz, 1, self.config.backbone_hidden_size), dtype = torch.half, device = dev)
            st.ids_out = torch.zeros((bsz, window), dtype = torch.long, device = dev)
            st.pos = torch.zeros((bsz,), dtype = torch.int32, device = dev)
            st.prev = torch.zeros((bsz,), dtype = torch.int32, device = dev)
            st.bt = torch.zeros(tuple(bt.shape), dtype = torch.int32, device = dev)
            st.swa_bt = torch.zeros((bsz, pps), dtype = torch.int32, device = dev)
            st.swa_sl = torch.zeros((bsz,), dtype = torch.int32, device = dev)
            st.params = {
                "g4a_graph": True,
                "outer_graph": True,
                "g4a_base_seqlens": st.pos,
                "g4a_prev_seqlens": st.prev,
                "block_table": st.bt,
                "g4a_swa_bt": st.swa_bt,
                "g4a_swa_sl": st.swa_sl,
                "target_cache": params["target_cache"],
                "target_recurrent_states": rsg,
            }
            st.graph = None

        # Round inputs
        sl = params["cache_seqlens"][:bsz]
        host = torch.empty((4, bsz), dtype = torch.int32)
        host[0].copy_(sl)
        host[1].copy_(sl - 1)
        for b, rs in enumerate(rsg):
            host[2, b] = rs.position - rs.window_beg - 1
        swa_bt = torch.tensor([[rs.slot * pps + j for j in range(pps)] for rs in rsg], dtype = torch.int32)
        st.params["target_recurrent_states"] = rsg
        st.ids0.copy_(ids, non_blocking = True)
        st.h0.copy_(hidden.view(st.h0.shape))
        st.pos.copy_(host[0], non_blocking = False)
        st.prev.copy_(host[1])
        st.swa_sl.copy_(host[2])
        st.swa_bt.copy_(swa_bt)
        st.bt.copy_(bt[:bsz])

        if fresh:
            try:
                # Eager warm-up on a side stream (cubin builds, cuBLAS handles), then capture
                s = torch.cuda.Stream(dev)
                s.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(s):
                    self._round_body(st)
                torch.cuda.current_stream(dev).wait_stream(s)
                torch.cuda.synchronize(dev)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._round_body(st)
                st.graph = g
                self._graphs[key] = st
            except Exception as e:
                print(f" !! Gemma 4 assistant: graph drafting unavailable, using the per-step loop: {e}")
                self._round_failed = True
                torch.cuda.synchronize(dev)
                return None

        st.graph.replay()
        return st.ids_out
