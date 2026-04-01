"""
从 Volcano (Mistral-7B) 各层 self-attention 的 KV cache 中导出当前前向步新增的 K/V。

张量语义与 HF DynamicCache 一致：[batch, num_key_value_heads, seq_chunk, head_dim]；
Mistral-7B 典型为 8 × 128（GQA）。保存为 numpy，按步、按层堆叠。
"""
from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


def build_prefill_modality_mask(
    input_ids: torch.LongTensor,
    image_token_id: int,
    num_image_tokens: int,
) -> List[int]:
    """将 <ImageHere> 单 token 展开为 num_image_tokens 个视觉位置（modal=1）。"""
    mask: List[int] = []
    row = input_ids[0] if input_ids.dim() == 2 else input_ids
    for tid in row.tolist():
        if tid == image_token_id:
            mask.extend([1] * int(num_image_tokens))
        else:
            mask.append(0)
    return mask


@dataclass
class _StepBuffer:
    q_len: int = 0
    positions: List[int] = field(default_factory=list)
    modal: List[int] = field(default_factory=list)
    layer_k: Dict[int, torch.Tensor] = field(default_factory=dict)
    layer_v: Dict[int, torch.Tensor] = field(default_factory=dict)


class MistralKVTracer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._hooks: List[Any] = []
        self._active = False
        self._prefill_modal: List[int] = []
        self._prefill_cursor = 0
        self._pending_modal: Deque[List[int]] = deque()
        self._position_offset = 0
        self._current_step: Optional[_StepBuffer] = None
        self._num_layers = 0
        self._sample_meta: Dict[str, Any] = {}
        self.steps: List[Dict[str, Any]] = []

    def set_prefill_modality(self, mask: List[int]) -> None:
        self._prefill_modal = list(mask)
        self._prefill_cursor = 0

    def push_modal_override(self, modal_per_pos: List[int]) -> None:
        self._pending_modal.append(list(modal_per_pos))

    def start_sample(self, meta: Dict[str, Any]) -> None:
        self._sample_meta = dict(meta)
        self.steps = []
        self._prefill_cursor = 0
        self._pending_modal.clear()
        self._position_offset = 0
        self._current_step = None

    def register(self, model: nn.Module) -> None:
        self.unregister()
        base, layers, n_layers = _resolve_decoder_layers(model)
        self._num_layers = n_layers
        for li, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            if getattr(attn, "layer_idx", None) is None:
                attn.layer_idx = li  # type: ignore[attr-defined]
            h = attn.register_forward_hook(self._make_attn_hook(li, n_layers), with_kwargs=False)
            self._hooks.append(h)

    def unregister(self) -> None:
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()

    def _begin_step(self, q_len: int, args: tuple) -> None:
        position_ids = args[2] if len(args) > 2 else None
        if position_ids is not None:
            pos = position_ids[0].tolist()
        else:
            pos = [self._position_offset + i for i in range(q_len)]

        if self._pending_modal:
            pending = self._pending_modal.popleft()
            if len(pending) == q_len:
                modal = pending
            elif len(pending) > q_len:
                modal = pending[:q_len]
            else:
                modal = pending + [0] * (q_len - len(pending))
        else:
            modal = self._consume_modal(q_len)

        self._current_step = _StepBuffer(q_len=q_len, positions=pos, modal=modal)
        if pos:
            self._position_offset = max(pos) + 1
        else:
            self._position_offset += q_len

    def _consume_modal(self, q_len: int) -> List[int]:
        if self._prefill_cursor < len(self._prefill_modal):
            take = min(q_len, len(self._prefill_modal) - self._prefill_cursor)
            chunk = self._prefill_modal[self._prefill_cursor : self._prefill_cursor + take]
            self._prefill_cursor += take
            if take < q_len:
                chunk = chunk + [0] * (q_len - take)
            return chunk
        return [0] * q_len

    def _finalize_step(self) -> None:
        if self._current_step is None:
            return
        buf = self._current_step
        if len(buf.layer_k) < self._num_layers:
            return
        rec: Dict[str, Any] = {
            "positions": buf.positions,
            "modal": buf.modal,
            "q_len": buf.q_len,
        }
        ks = []
        vs = []
        for li in range(self._num_layers):
            if li not in buf.layer_k:
                return
            ks.append(buf.layer_k[li][0].float().cpu().numpy())
            vs.append(buf.layer_v[li][0].float().cpu().numpy())
        rec["k_layers"] = np.stack(ks, axis=0)
        rec["v_layers"] = np.stack(vs, axis=0)
        self.steps.append(rec)
        self._current_step = None

    def _make_attn_hook(self, layer_idx: int, num_layers: int) -> Callable:

        def hook(module, args, output):
            if not self.enabled or not self._active:
                return
            if output is None or len(output) < 3:
                return
            hidden_states = args[0]
            if hidden_states is None:
                return
            q_len = int(hidden_states.shape[1])
            past = output[2]
            if past is None or not hasattr(past, "key_cache"):
                return

            li = getattr(module, "layer_idx", layer_idx)
            if li >= len(past.key_cache):
                return
            k_full = past.key_cache[li]
            v_full = past.value_cache[li]
            if k_full.shape[-2] < q_len:
                return
            k_new = k_full[:, :, -q_len:, :].detach().contiguous()
            v_new = v_full[:, :, -q_len:, :].detach().contiguous()

            if li == 0:
                self._begin_step(q_len, args)

            if self._current_step is None:
                return
            self._current_step.layer_k[li] = k_new
            self._current_step.layer_v[li] = v_new

            if li == num_layers - 1:
                self._finalize_step()

        return hook

    def __enter__(self):
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self._active = False
        return False


def _resolve_decoder_layers(model: nn.Module):
    layers = None
    inner = model
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        layers = inner.model.layers
    if layers is None and hasattr(inner, "base_model"):
        bm = inner.base_model
        if hasattr(bm, "model") and hasattr(bm.model, "layers"):
            layers = bm.model.layers
    if layers is None:
        raise ValueError("未找到 decoder layers（尝试 model.model.layers / model.base_model.model.layers）。")
    n = len(layers)
    return inner, layers, n


def patch_volcano_generators_for_modal(model: nn.Module, tracer: MistralKVTracer) -> None:
    if getattr(model, "_kv_trace_patched", False):
        return
    if not hasattr(model, "generate_box") or not hasattr(model, "generate_image"):
        return

    orig_box = model.generate_box
    orig_img = model.generate_image

    def gen_box_wrapped(*a, **kw):
        out = orig_box(*a, **kw)
        embeds = out[0]
        q = embeds.shape[1]
        tracer.push_modal_override([0] + [1] * (q - 1))
        return out

    def gen_img_wrapped(*a, **kw):
        out = orig_img(*a, **kw)
        embeds = out[0]
        q = embeds.shape[1]
        tracer.push_modal_override([1] * q)
        return out

    model.generate_box = gen_box_wrapped
    model.generate_image = gen_img_wrapped
    model._kv_trace_patched = True
    model._kv_trace_orig_generate_box = orig_box
    model._kv_trace_orig_generate_image = orig_img


def unpatch_volcano_generators(model: nn.Module) -> None:
    if getattr(model, "_kv_trace_patched", False):
        model.generate_box = model._kv_trace_orig_generate_box
        model.generate_image = model._kv_trace_orig_generate_image
        delattr(model, "_kv_trace_patched")


def save_sample_kv(
    out_dir: str,
    tracer: MistralKVTracer,
    meta: Dict[str, Any],
    dtype_save: str = "float16",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    meta_out = {
        **meta,
        "num_steps": len(tracer.steps),
        "tensor_layout_note": (
            "k_layers / v_layers: (num_layers, num_kv_heads, q_len, head_dim)，"
            "与 HF DynamicCache 中物理 KV 头一致（GQA 下未 repeat 到 query 头）。"
            "positions / modal 长度均为 q_len，对应本步新增的 token。"
        ),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    steps_dir = os.path.join(out_dir, "steps")
    os.makedirs(steps_dir, exist_ok=True)

    for si, step in enumerate(tracer.steps):
        k = step["k_layers"]
        v = step["v_layers"]
        if dtype_save == "float16":
            k = np.asarray(k, dtype=np.float16)
            v = np.asarray(v, dtype=np.float16)
        np.savez_compressed(
            os.path.join(steps_dir, f"step_{si:05d}.npz"),
            k_layers=k,
            v_layers=v,
            positions=np.array(step["positions"], dtype=np.int32),
            modal=np.array(step["modal"], dtype=np.int8),
            q_len=np.int32(step["q_len"]),
        )


def save_sample_kv_hierarchical(
    out_dir: str,
    tracer: MistralKVTracer,
    meta: Dict[str, Any],
    dtype_save: str = "float16",
    trajectory_idx: int = 0,
) -> None:
    """
    按 layer_* / kv_* / trajectory_* / token_* / kv_pos_modal.npz 写出每条 token 的 (k,v,pos,modal)。

    目录：out_dir/layer_{L:02d}/kv_{H}/trajectory_{T}/token_{global:06d}/kv_pos_modal.npz
    - k, v: (head_dim,) 例如 128，dtype 由 dtype_save 决定
    - pos: (1,) int32
    - modal: (1,) int8

    meta 须含 num_hidden_layers, num_key_value_heads, head_dim（与张量形状一致）。
    """
    L = int(meta["num_hidden_layers"])
    Hkv = int(meta["num_key_value_heads"])
    D = int(meta["head_dim"])

    np_f = np.float16 if dtype_save == "float16" else np.float32
    if dtype_save not in ("float16", "float32"):
        raise ValueError("dtype_save 须为 float16 或 float32")

    os.makedirs(out_dir, exist_ok=True)

    token_global = 0
    for step in tracer.steps:
        K = np.asarray(step["k_layers"])
        V = np.asarray(step["v_layers"])
        positions = step["positions"]
        modals = step["modal"]
        qlen = int(step["q_len"])
        if K.shape != (L, Hkv, qlen, D) or V.shape != K.shape:
            raise ValueError(
                f"本步 k/v 形状 {K.shape} 与 meta (layers={L}, kv_heads={Hkv}, q_len={qlen}, head_dim={D}) 不一致"
            )
        if len(positions) != qlen or len(modals) != qlen:
            raise ValueError("positions / modal 长度须等于 q_len")

        for j in range(qlen):
            pos_arr = np.array([int(positions[j])], dtype=np.int32)
            modal_arr = np.array([int(modals[j])], dtype=np.int8)
            for li in range(L):
                for hi in range(Hkv):
                    token_dir = os.path.join(
                        out_dir,
                        f"layer_{li:02d}",
                        f"kv_{hi}",
                        f"trajectory_{trajectory_idx}",
                        f"token_{token_global:06d}",
                    )
                    os.makedirs(token_dir, exist_ok=True)
                    k_vec = np.asarray(K[li, hi, j, :], dtype=np_f)
                    v_vec = np.asarray(V[li, hi, j, :], dtype=np_f)
                    np.savez_compressed(
                        os.path.join(token_dir, "kv_pos_modal.npz"),
                        k=k_vec,
                        v=v_vec,
                        pos=pos_arr,
                        modal=modal_arr,
                    )
            token_global += 1

    meta_out = {
        **meta,
        "num_steps": len(tracer.steps),
        "num_tokens": token_global,
        "trajectory_idx": int(trajectory_idx),
        "kv_layout": "hierarchical",
        "path_pattern": (
            "layer_{layer:02d}/kv_{kv_head}/trajectory_{traj}/token_{tok:06d}/kv_pos_modal.npz"
        ),
        "arrays_per_file": {
            "k": f"(head_dim,), {dtype_save}",
            "v": f"(head_dim,), {dtype_save}",
            "pos": "(1,) int32",
            "modal": "(1,) int8",
        },
        "gqa_note": (
            "KV cache 仅含 num_key_value_heads 个物理头（如 8），维度 head_dim=hidden/num_attention_heads（如 128）；"
            "与 query 头数 32 不对等，若需对齐 32 路 Q 头请在训练侧对 K/V 做 repeat_kv 再使用。"
        ),
        "rope_note": (
            "k、v 与 HF DynamicCache 一致，一般为应用 RoPE 后的 key/value states。"
        ),
        "token_order_note": (
            "token_{global} 按前向步顺序拼接：含 prefill 一步内多 token（q_len>1）及 decode 每步通常 q_len=1。"
        ),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
