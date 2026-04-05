"""
解码每步 logits 的 softmax 熵 H = -sum_v p(v) log p(v)，并可视化。

与 transformers generate(output_scores=True) 返回的 scores 对齐：每项为一步的 (batch, vocab) logits。
"""
from __future__ import annotations

import json
import math
import unicodedata
from typing import Any, Sequence

import torch
import torch.nn.functional as F


def softmax_entropy_from_logits(logits: torch.Tensor) -> float:
    """logits: (vocab,) 或 (1, vocab)，在末维做 softmax 后 Shannon 熵（nat，自然对数）。"""
    x = logits.detach().float().view(-1)
    log_p = F.log_softmax(x, dim=-1)
    p = log_p.exp()
    h = -(p * log_p).sum()
    return float(h.item())


def entropy_and_prob_chosen_from_logits(
    logits: torch.Tensor,
    chosen_token_id: int,
) -> tuple[float, float, float]:
    """
    Single softmax over vocab: Shannon entropy (nat), p(chosen), log p(chosen) (nat).
    If chosen_token_id out of range, prob and log_prob are nan.
    """
    x = logits.detach().float().view(-1)
    v = int(x.shape[0])
    log_p = F.log_softmax(x, dim=-1)
    pr = log_p.exp()
    h = -(pr * log_p).sum()
    if 0 <= int(chosen_token_id) < v:
        lp = float(log_p[chosen_token_id].item())
        prob = float(math.exp(lp))
    else:
        lp = float("nan")
        prob = float("nan")
    return float(h.item()), prob, lp


def build_entropy_trace(
    tokenizer,
    sequences: torch.LongTensor,
    input_token_len: int,
    scores: Sequence[torch.Tensor] | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    将 scores 与 sequences[:, input_token_len:] 按步对齐（取较短长度）。
    返回 (trace 列表, 若长度不一致则返回警告文案否则 None)。
    """
    if scores is None or len(scores) == 0:
        return [], "No generation scores (enable output_scores=True on generate)."

    row = sequences[0] if sequences.dim() == 2 else sequences
    gen_ids = row[input_token_len:].tolist()
    n_sc = len(scores)
    n_tok = len(gen_ids)
    n = min(n_sc, n_tok)
    warn = None
    if n_sc != n_tok:
        warn = (
            f"len(scores)={n_sc} != num generated tokens={n_tok}; "
            f"aligned first {n} steps (VoCoT may inject non-standard LM steps at EOC/BOI)."
        )

    trace: list[dict[str, Any]] = []
    for i in range(n):
        logits = scores[i][0]
        tid = int(gen_ids[i])
        e, prob, log_prob = entropy_and_prob_chosen_from_logits(logits, tid)
        tok = tokenizer.decode([tid], skip_special_tokens=False)
        trace.append(
            {
                "step": i + 1,
                "token_id": tid,
                "token": tok,
                "entropy": e,
                "prob": prob,
                "log_prob": log_prob,
            }
        )
    return trace, warn


def _short_token_label(s: str, max_len: int = 16) -> str:
    s = repr(s)[1:-1] if s else ""
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _norm_decode_tok(tok: str | None) -> str:
    if tok is None:
        return ""
    return str(tok).strip()


def coordinate_inner_trace_indices(trace: list[dict[str, Any]]) -> set[int]:
    """
    Indices of steps whose decoded string lies **between** a ``<coor>`` and the next ``</coor>``
    (exclusive of the tag tokens themselves): i.e. bbox number / comma / dot stream tokens.
    Unclosed ``<coor>`` treats everything after it until end of trace as inner.
    """
    inner: set[int] = set()
    in_span = False
    for i, row in enumerate(trace):
        t = _norm_decode_tok(row.get("token"))
        if not in_span:
            if t == "<coor>":
                in_span = True
            continue
        if t == "</coor>":
            in_span = False
            continue
        inner.add(i)
    return inner


def _is_whitespace_or_punct_only_token(tok: str | None) -> bool:
    """
    True => skip for "low entropy half": empty/only-space, or only punctuation/separators
    (no Unicode letter L* nor digit N*). Catches ASCII/comma/period/? and CJK 。， etc.
    """
    if tok is None:
        return True
    s = str(tok)
    if not s.strip():
        return True
    for ch in s:
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N"):
            return False
    return True


def content_token_trace_indices(trace: list[dict[str, Any]]) -> list[int]:
    """
    Steps that participate in entropy **ranking / tiers** (and low-half selection):
    must contain a Unicode letter or digit (L*/N*), and must not be inside ``<coor>...</coor>``
    coordinate payload (nor the tag lines themselves — tags are not in ``inner``).
    """
    inner = coordinate_inner_trace_indices(trace)
    return [
        i
        for i in range(len(trace))
        if i not in inner and not _is_whitespace_or_punct_only_token(trace[i].get("token"))
    ]


def lowest_entropy_step_indices(
    trace: list[dict[str, Any]],
    fraction: float = 0.5,
) -> list[int]:
    """
    Lowest-entropy fraction among **eligible** steps only: not punct/space-only, not
    ``<coor>``…``</coor>`` inner coordinate tokens. ``fraction`` applies to eligible count.
    """
    content = content_token_trace_indices(trace)
    n = len(content)
    if n == 0:
        return []
    k = max(1, int(n * float(fraction) + 0.5))
    k = min(k, n)
    ranked = sorted(content, key=lambda i: trace[i]["entropy"])
    return sorted(ranked[:k])


def low_entropy_half_rows(
    trace: list[dict[str, Any]],
    fraction: float = 0.5,
) -> list[dict[str, Any]]:
    """Rows for the lowest-entropy fraction, sorted by entropy ascending (then step)."""
    idx = lowest_entropy_step_indices(trace, fraction=fraction)
    rows = [dict(trace[i]) for i in idx]
    rows.sort(key=lambda r: (r["entropy"], r["step"]))
    return rows


def concat_trace_tokens_in_step_order(
    trace: list[dict[str, Any]],
    trace_indices: list[int],
) -> str:
    """Join decoded `token` strings for given trace row indices in ascending step / time order."""
    parts: list[str] = []
    for i in sorted(trace_indices):
        t = trace[i].get("token")
        parts.append("" if t is None else str(t))
    return "".join(parts)


def entropy_high_low_tier_concat_in_step_order(
    trace: list[dict[str, Any]],
    high_entropy_fraction: float = 0.2,
) -> tuple[str, str]:
    """
    Among **eligible** steps (same filter as ``content_token_trace_indices``): split by entropy rank
    into highest ``high_entropy_fraction`` vs the rest. Concatenate each side's ``token`` strings in
    **generation step order**. Ineligible steps do not appear in either string.
    Returns (highest_tier_concat, lowest_tier_concat).
    """
    eligible = content_token_trace_indices(trace)
    n = len(eligible)
    if n == 0:
        return "", ""
    hf = max(0.0, min(1.0, float(high_entropy_fraction)))
    k_hi = max(0, min(n, round(n * hf)))
    k_lo = n - k_hi
    by_entropy = sorted(eligible, key=lambda i: trace[i]["entropy"])
    low_ix = by_entropy[:k_lo]
    high_ix = by_entropy[k_lo:]
    low_s = concat_trace_tokens_in_step_order(trace, low_ix)
    high_s = concat_trace_tokens_in_step_order(trace, high_ix)
    return high_s, low_s


def format_low_entropy_analysis(
    trace: list[dict[str, Any]],
    fraction: float = 0.5,
    max_lines: int | None = None,
) -> str:
    """Plain-text block for stderr / appendix after model output."""
    rows = low_entropy_half_rows(trace, fraction=fraction)
    if not rows:
        return ""
    n_all = len(trace)
    n_content = len(content_token_trace_indices(trace))
    head = (
        f"--- Lowest-entropy {fraction:.0%} among eligible tokens "
        f"(picked {len(rows)}/{n_content} eligible steps; full trace {n_all}; "
        "excl. punct/space-only & <coor>...</coor> inner coords) ---\n"
        "Interpretation: lower H => narrower next-token distribution => model more confident.\n"
    )
    lines = [head]
    for j, r in enumerate(rows):
        if max_lines is not None and j >= max_lines:
            lines.append(f"... ({len(rows) - max_lines} more, see entropy JSON key low_entropy_half)")
            break
        tok = repr(r["token"])
        p = r.get("prob", float("nan"))
        if isinstance(p, float) and not math.isnan(p) and p < 1e-8:
            p_s = f"{p:.2e}"
        else:
            p_s = f"{p:.6g}"
        lines.append(
            f"  step={r['step']:<5}  H={r['entropy']:.5f}  p={p_s}  id={r['token_id']:<6}  tok={tok}"
        )
    return "\n".join(lines) + "\n"


def save_entropy_plot(
    trace: list[dict[str, Any]],
    out_path: str,
    top_k: int = 10,
    title: str | None = None,
    low_entropy_fraction: float = 0.5,
    low_entropy_label_max: int = 50,
) -> None:
    """Line chart: high-entropy annotations (red); low-entropy half marked (green markers + labels)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not trace:
        raise ValueError("empty trace, cannot plot")

    x = [r["step"] for r in trace]
    y = [r["entropy"] for r in trace]

    fig, ax = plt.subplots(figsize=(max(8, len(x) * 0.12), 4.5))
    ax.plot(x, y, color="#2563eb", linewidth=1.2, marker="o", markersize=2)
    ax.set_xlabel("Generation step (output_scores index)")
    ax.set_ylabel("Entropy (nat)")
    ax.set_title(title or "Per-step softmax entropy (Shannon, LM head logits)")
    ax.grid(True, alpha=0.3)

    low_ix = set(lowest_entropy_step_indices(trace, fraction=low_entropy_fraction))
    for i in low_ix:
        ax.plot(
            x[i],
            y[i],
            marker="o",
            markersize=6,
            markerfacecolor="none",
            markeredgewidth=1.8,
            markeredgecolor="#15803d",
        )

    order_hi = sorted(range(len(y)), key=lambda i: y[i], reverse=True)
    picked_hi = set()
    for i in order_hi:
        if len(picked_hi) >= top_k:
            break
        picked_hi.add(i)

    for i in sorted(picked_hi):
        lab = _short_token_label(trace[i]["token"])
        ax.annotate(
            lab,
            xy=(x[i], y[i]),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=7,
            color="#b91c1c",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.85),
        )

    low_by_h = sorted(low_ix, key=lambda i: trace[i]["entropy"])
    for rank, i in enumerate(low_by_h[: max(0, low_entropy_label_max)]):
        lab = _short_token_label(trace[i]["token"])
        ax.annotate(
            lab,
            xy=(x[i], y[i]),
            xytext=(6, -10 - 7 * (rank % 6)),
            textcoords="offset points",
            fontsize=6,
            color="#15803d",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#dcfce7", alpha=0.9),
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_entropy_json(trace: list[dict[str, Any]], meta: dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "trace": trace}, f, ensure_ascii=False, indent=2)
