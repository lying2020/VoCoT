"""
子集 GQA 官方指标（eval.py）支持。

官方 eval.py 要求 questions 中每条 balanced 题在 predictions 里都有条目；子集推理时改为
写入「与 GQADataset 同序」截取的前 N 题的 questions/choices JSON，并对 eval.py 打一行补丁，
使 consistency 在 entailed 题不在子集内时跳过（避免 KeyError）。
"""
from __future__ import annotations

import json
import os
import shutil


def _val_balanced_paths(gqa_root: str) -> tuple[str, str]:
    q = os.path.join(gqa_root, "questions1.2", "val_balanced_questions.json")
    c = os.path.join(gqa_root, "eval", "val_choices.json")
    return q, c


def prepare_val_balanced_subset_files(
    gqa_root: str, max_samples: int, out_dir: str
) -> tuple[str, str, int]:
    """
    按 JSON 字典键顺序（与 GQADataset / json.load 一致）取前 N 题，写出过滤后的
    val_balanced_questions_subset.json 与 val_choices_subset.json。

    返回 (questions_path, choices_path, n_effective)。
    """
    if max_samples <= 0:
        raise ValueError("max_samples 须为正整数")

    q_full, c_full = _val_balanced_paths(gqa_root)
    with open(q_full, encoding="utf-8") as f:
        questions = json.load(f)
    with open(c_full, encoding="utf-8") as f:
        choices = json.load(f)

    keys = list(questions.keys())
    n_eff = min(max_samples, len(keys))
    keys = keys[:n_eff]

    q_sub = {k: questions[k] for k in keys}
    missing_c = [k for k in keys if k not in choices]
    if missing_c:
        raise KeyError(
            f"val_choices.json 缺少 {len(missing_c)} 道题的条目，示例: {missing_c[:3]}"
        )
    c_sub = {k: choices[k] for k in keys}

    os.makedirs(out_dir, exist_ok=True)
    q_out = os.path.join(out_dir, "val_balanced_questions_subset.json")
    c_out = os.path.join(out_dir, "val_choices_subset.json")
    with open(q_out, "w", encoding="utf-8") as f:
        json.dump(q_sub, f)
    with open(c_out, "w", encoding="utf-8") as f:
        json.dump(c_sub, f)

    return q_out, c_out, n_eff


def write_subset_safe_eval_script(gqa_root: str, cache_dir: str) -> str:
    """
    复制 GQA_Bench/eval/eval.py 到 cache_dir，并打子集安全补丁（仅一行）。
    若已存在且内容已包含补丁则复用。

    返回可执行的 eval 脚本绝对路径。
    """
    src = os.path.join(gqa_root, "eval", "eval.py")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"未找到官方 eval: {src}")

    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, "eval_subset_safe.py")

    with open(src, encoding="utf-8") as f:
        text = f.read()

    marker = "eid in questions and eid in predictions"
    if marker not in text:
        old = 'inferredQuestions = [eid for eid in question["entailed"] if eid != questionId]'
        new = (
            'inferredQuestions = [eid for eid in question["entailed"] '
            'if eid != questionId and eid in questions and eid in predictions]'
        )
        if old not in text:
            raise RuntimeError(
                "GQA eval.py 与 VoCoT 子集补丁不匹配（未找到 updateConsistency 中的 inferredQuestions 行）。"
                "请检查 GQA_Bench 版本是否与官方一致。"
            )
        text = text.replace(old, new, 1)

    with open(dst, "w", encoding="utf-8") as f:
        f.write(text)

    shutil.copymode(src, dst)
    return dst
