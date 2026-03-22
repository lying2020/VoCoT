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
  （manifest.json、results.json、images/*.jpg）；可用 --output_dir / --is_save_results 调整。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PIL import Image

from model.load_model import infer, load_model
from project import gqa_bench_path, volcano_7b_luoruipu1_path


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
        "--is_save_results",
        type=bool,
        default=False,
        help="不写 manifest/results/图片副本，仅终端或 --jsonl 输出",
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
    if args.is_save_results:
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    if not args.jsonl:
        print(f"# model_path: {model_path}", file=sys.stderr)
        print(f"# questions: {qpath}", file=sys.stderr)
        print(f"# 本题数: {len(run_keys)}", file=sys.stderr)
        if args.is_save_results:
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

        rel_saved = f"images/{qid}.jpg"
        dst_img = os.path.join(out_dir, rel_saved) if not args.is_save_results else None
        if not args.is_save_results and not args.no_copy_images:
            shutil.copy2(img_path, dst_img)

        rec = {
            "questionId": qid,
            "imageId": image_id,
            "source_image_path": img_path,
            "question": question,
            "prediction": text,
        }
        if "answer" in item:
            rec["answer"] = item["answer"]
        if not args.is_save_results:
            if args.no_copy_images:
                rec["saved_image"] = None
            else:
                rec["saved_image"] = rel_saved

        results_rows.append(rec)

        if args.jsonl:
            print(json.dumps(rec, ensure_ascii=False))
        else:
            print("-" * 60)
            print(f"questionId: {qid}")
            print(f"imageId:    {image_id}")
            print(f"image:      {img_path}")
            print(f"question:   {question}")
            print("-" * 60)
            print(text)
            print()

    if not args.is_save_results:
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
        }
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results_rows, f, ensure_ascii=False, indent=2)
        if not args.jsonl:
            print(f"# 已写入 manifest.json、results.json", file=sys.stderr)
            if not args.no_copy_images:
                print(f"# 图片副本目录: {os.path.join(out_dir, 'images')}", file=sys.stderr)


if __name__ == "__main__":
    main()
