#!/usr/bin/env python3
"""
使用 GQA_Bench 中的图片与问题做本地推理测试（不跑完整评测、不写 benchmark JSON）。

在 VoCoT 仓库根目录执行:
  python run_gqa_inference_demo.py
  python run_gqa_inference_demo.py --n 5 --start 100
  python run_gqa_inference_demo.py --split testdev --question_id 07333408
  python run_gqa_inference_demo.py --no_cot --max_new_tokens 256

默认数据根目录: project.gqa_bench_path；默认模型: project.volcano_7b_luoruipu1_path（与 run_inference_demo 一致）。

默认将本次用到的题目、预测与图片副本写入:
  output/gqa/<模型目录名>/inference_demo/
  （manifest.json、results.json、images/*.jpg）；可用 --output_dir、--no_save、--no_copy_images 调整。

  默认另存每步 softmax 熵（H=-Σ p log p，自然对数 nat）与曲线图：
    entropy/<questionId>_entropy.png 、 *_entropy.json ；可用 --no_entropy 关闭。
  图中绿色圈/标签为熵最低的一批 token（默认在「去掉纯空格/标点」后的 token 上取最低 50%）；题末打印同批分析；JSON meta.low_entropy_half。
  results.json 每条含 entropy_highest_20pct_concat_in_order、entropy_lowest_80pct_concat_in_order：
    仅在「非纯标点/空格、且非 <coor>...</coor> 内部坐标串」的 step 上划分高/低熵比例后按 step 顺序拼接。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PIL import Image

from model.load_model import infer, infer_with_generation_scores, load_model
from project import gqa_bench_path, volcano_7b_luoruipu1_path
from utils.token_entropy_viz import (
    build_entropy_trace,
    content_token_trace_indices,
    entropy_high_low_tier_concat_in_step_order,
    format_low_entropy_analysis,
    low_entropy_half_rows,
    save_entropy_json,
    save_entropy_plot,
)


def _questions_path(gqa_root: str, split: str) -> str:
    if split == "val":
        name = "val_balanced_questions.json"
    elif split == "testdev":
        name = "testdev_balanced_questions.json"
    else:
        raise ValueError("split 须为 val 或 testdev")
    return os.path.join(gqa_root, "questions1.2", name)


def _image_path(gqa_root: str, image_id: str) -> str:
    return os.path.join(gqa_root, "images", "images", f"{image_id}.jpg")


def _resolve_model_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.path.isdir(volcano_7b_luoruipu1_path):
        return volcano_7b_luoruipu1_path
    return "luoruipu1/Volcano-7b"


def _default_output_dir(model_path: str) -> str:
    """与 run_gqa_bench 的 output/gqa/<模型名>/ 对齐，子目录为 inference_demo。"""
    ap = os.path.abspath(model_path)
    store = os.path.basename(ap.rstrip(os.sep))
    if not store or store in (".", ".."):
        store = "model"
    return os.path.join(_REPO_ROOT, "output", "gqa", store, "inference_demo")


def main() -> None:
    p = argparse.ArgumentParser(description="GQA 图片 + 问题 本地推理（非完整评测）")
    p.add_argument("--gqa_root", type=str, default=gqa_bench_path, help="GQA_Bench 根目录")
    p.add_argument("--split", type=str, default="val", choices=("val", "testdev"))
    p.add_argument(
        "--n",
        type=int,
        default=10,
        help="连续跑多少道题（与评测集相同键顺序；与 --question_id 互斥）",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="从第几条题开始（0-based，按 questions JSON 键顺序）",
    )
    p.add_argument(
        "--question_id",
        type=str,
        default=None,
        help="只跑指定 questionId（覆盖 --n / --start）",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="在取题前先打乱键顺序（与 --seed 同用可复现）",
    )
    p.add_argument("--seed", type=int, default=None, help="随机种子（仅 --shuffle 时有意义）")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--precision", type=str, default="fp16", choices=("fp16", "bf16", "fp32"))
    p.add_argument("--no_cot", action="store_true", help="关闭 VoCoT/grounding+CoT 提示")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--jsonl",
        action="store_true",
        help="每行输出一条 JSON（questionId、question、prediction 等）",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="评测结果目录；默认 output/gqa/<模型目录名>/inference_demo（仓库根下绝对路径）",
    )
    p.add_argument(
        "--no_save",
        action="store_true",
        help="不写 manifest.json / results.json / 图片副本，仅终端或 --jsonl 输出",
    )
    p.add_argument(
        "--no_copy_images",
        action="store_true",
        help="仍写 manifest 与 results.json，但不复制图片（仅记录 source_image_path）",
    )
    p.add_argument(
        "--no_entropy",
        action="store_true",
        help="不计算/保存 token softmax 熵与折线图（略省显存与时间）",
    )
    p.add_argument(
        "--entropy_top_k",
        type=int,
        default=10,
        help="折线图上标注熵最高的前 K 个 token 文本",
    )
    p.add_argument(
        "--low_entropy_fraction",
        type=float,
        default=0.5,
        help="熵最低的比例（0–1），在排除纯空格/纯标点片段后计数的 token 上取值；用于绿标注、终端与 meta.low_entropy_half",
    )
    p.add_argument(
        "--low_entropy_label_max",
        type=int,
        default=50,
        help="图中对最低熵集合最多打多少个绿色文字标签（其余仅绿色圈）",
    )
    p.add_argument(
        "--low_entropy_report_max_lines",
        type=int,
        default=120,
        help="每题终端附录最多列多少行最低熵 token（JSON 仍完整）",
    )

    args = p.parse_args()

    qpath = _questions_path(args.gqa_root, args.split)
    if not os.path.isfile(qpath):
        raise FileNotFoundError(f"找不到问题文件: {qpath}")

    with open(qpath, encoding="utf-8") as f:
        all_q = json.load(f)
    keys = list(all_q.keys())

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(keys)

    if args.question_id is not None:
        if args.question_id not in all_q:
            raise KeyError(f"questionId 不在当前 split 中: {args.question_id}")
        run_keys = [args.question_id]
    else:
        if args.start < 0 or args.start >= len(keys):
            raise IndexError(f"--start={args.start} 越界（共 {len(keys)} 题）")
        end = min(args.start + max(args.n, 0), len(keys))
        run_keys = keys[args.start:end]
        if not run_keys:
            raise ValueError("没有可跑的题目（检查 --n / --start）")

    model_path = _resolve_model_path(args.model_path)
    out_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else _default_output_dir(model_path)
    )
    if not args.no_save:
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    if not args.no_save and not args.no_entropy:
        os.makedirs(os.path.join(out_dir, "entropy"), exist_ok=True)

    if not args.jsonl:
        print(f"# model_path: {model_path}", file=sys.stderr)
        print(f"# questions: {qpath}", file=sys.stderr)
        print(f"# 本题数: {len(run_keys)}", file=sys.stderr)
        if not args.no_save:
            print(f"# 输出目录: {out_dir}", file=sys.stderr)

    model, preprocessor = load_model(model_path, device=args.device, precision=args.precision)

    results_rows: list[dict] = []

    for qid in run_keys:
        item = all_q[qid]
        question = item["question"]
        image_id = str(item["imageId"])
        img_path = _image_path(args.gqa_root, image_id)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"找不到图像: {img_path}")

        image = Image.open(img_path).convert("RGB")
        entropy_plot_rel = None
        entropy_json_rel = None
        alignment_warn: str | None = None
        trace: list = []

        if args.no_entropy or args.no_save:
            out = infer(
                model,
                preprocessor,
                image,
                question,
                cot=not args.no_cot,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            text = out[0] if isinstance(out, (list, tuple)) else out
        else:
            pred_list, _out_imgs, txt_ids, scores, in_len = infer_with_generation_scores(
                model,
                preprocessor,
                image,
                question,
                cot=not args.no_cot,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            text = pred_list[0] if isinstance(pred_list, (list, tuple)) else pred_list
            trace, alignment_warn = build_entropy_trace(
                preprocessor.tokenizer,
                txt_ids,
                in_len,
                scores,
            )
            if not args.no_save and trace:
                entropy_dir = os.path.join(out_dir, "entropy")
                plot_path = os.path.join(entropy_dir, f"{qid}_entropy.png")
                json_path = os.path.join(entropy_dir, f"{qid}_entropy.json")
                lf = max(0.01, min(1.0, float(args.low_entropy_fraction)))
                save_entropy_plot(
                    trace,
                    plot_path,
                    top_k=max(1, args.entropy_top_k),
                    title=f"questionId={qid}  softmax entropy (nat)",
                    low_entropy_fraction=lf,
                    low_entropy_label_max=max(0, args.low_entropy_label_max),
                )
                low_rows = low_entropy_half_rows(trace, fraction=lf)
                n_content = len(content_token_trace_indices(trace))
                meta = {
                    "questionId": qid,
                    "entropy_formula": "H=-sum_v p(v|context) log p(v), p=softmax(logits)",
                    "log_base": "natural (nat)",
                    "trace_fields": {
                        "prob": "p(chosen_token_id) from softmax over vocab at this generation step",
                        "log_prob": "ln(prob), same log base as entropy",
                    },
                    "low_entropy_half_filter": (
                        "Excluded before entropy ranking / 20-80% tiers: (1) no Unicode letter (L*) "
                        "nor digit (N*) — pure punctuation/whitespace/empty; (2) decoded tokens strictly "
                        "between <coor> and </coor> (bbox numeric stream); boundary tags themselves are not inner."
                    ),
                    "trace_steps_total": len(trace),
                    "low_entropy_content_steps": n_content,
                    "low_entropy_half_saved_count": len(low_rows),
                    "alignment_note": alignment_warn,
                    "low_entropy_fraction": lf,
                    "low_entropy_half": low_rows,
                }
                save_entropy_json(trace, meta, json_path)
                entropy_plot_rel = f"entropy/{qid}_entropy.png"
                entropy_json_rel = f"entropy/{qid}_entropy.json"
                if alignment_warn and not args.jsonl:
                    print(f"# [{qid}] {alignment_warn}", file=sys.stderr)

        rel_saved = f"images/{qid}.jpg"
        dst_img = os.path.join(out_dir, rel_saved)
        if not args.no_save and not args.no_copy_images:
            shutil.copy2(img_path, dst_img)

        rec = {
            "questionId": qid,
            "imageId": image_id,
            "source_image_path": img_path,
            "question": question,
            "prediction": text,
        }
        if trace:
            hi_concat, lo_concat = entropy_high_low_tier_concat_in_step_order(
                trace, high_entropy_fraction=0.2
            )
            rec["entropy_highest_20pct_concat_in_order"] = hi_concat
            rec["entropy_lowest_80pct_concat_in_order"] = lo_concat
        else:
            rec["entropy_highest_20pct_concat_in_order"] = ""
            rec["entropy_lowest_80pct_concat_in_order"] = ""
        if "answer" in item:
            rec["answer"] = item["answer"]
        if not args.no_save:
            rec["saved_image"] = None if args.no_copy_images else rel_saved
        if entropy_plot_rel is not None:
            rec["entropy_plot"] = entropy_plot_rel
            rec["entropy_json"] = entropy_json_rel

        results_rows.append(rec)

        if args.jsonl:
            print(json.dumps(rec, ensure_ascii=False))
            if trace:
                lf = max(0.01, min(1.0, float(args.low_entropy_fraction)))
                analysis = format_low_entropy_analysis(
                    trace,
                    fraction=lf,
                    max_lines=max(0, args.low_entropy_report_max_lines),
                )
                print(analysis, end="", file=sys.stderr)
        else:
            print("-" * 60)
            print(f"questionId: {qid}")
            print(f"imageId:    {image_id}")
            print(f"image:      {img_path}")
            print(f"question:   {question}")
            print("-" * 60)
            print(text)
            print()
            if trace:
                lf = max(0.01, min(1.0, float(args.low_entropy_fraction)))
                analysis = format_low_entropy_analysis(
                    trace,
                    fraction=lf,
                    max_lines=max(0, args.low_entropy_report_max_lines),
                )
                print(analysis, end="")

    if not args.no_save:
        ts = datetime.now(timezone.utc).isoformat()
        manifest = {
            "created_at_utc": ts,
            "output_dir": out_dir,
            "gqa_root": os.path.abspath(args.gqa_root),
            "split": args.split,
            "questions_file": os.path.abspath(qpath),
            "model_path": model_path,
            "device": args.device,
            "precision": args.precision,
            "no_cot": args.no_cot,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "start": args.start,
            "n": args.n,
            "question_id": args.question_id,
            "run_question_ids": run_keys,
            "num_questions": len(run_keys),
            "no_copy_images": args.no_copy_images,
            "no_entropy": args.no_entropy,
            "entropy_top_k": args.entropy_top_k,
            "low_entropy_fraction": args.low_entropy_fraction,
            "low_entropy_label_max": args.low_entropy_label_max,
            "low_entropy_report_max_lines": args.low_entropy_report_max_lines,
        }
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results_rows, f, ensure_ascii=False, indent=2)
        if not args.jsonl:
            print(f"# 已写入 manifest.json、results.json", file=sys.stderr)
            if not args.no_copy_images:
                print(f"# 图片副本目录: {os.path.join(out_dir, 'images')}", file=sys.stderr)
            if not args.no_entropy:
                print(
                    f"# token 熵曲线目录: {os.path.join(out_dir, 'entropy')}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
