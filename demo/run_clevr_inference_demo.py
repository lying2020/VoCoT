#!/usr/bin/env python3
"""
CLEVR 评测集本地推理试跑（与 CLEVRDataset / evaluate_benchmark 数据格式对齐，不跑全量 benchmark）。

默认使用 4-choice、按题型采样的 val 子集 JSON（与 config/datasets/eval/CLEVR_val1k*.yaml 一致），
图片目录为 CLEVR_v1.0/images/val/。

在 VoCoT 仓库根目录执行:
  python run_clevr_inference_demo.py
  python run_clevr_inference_demo.py --n 8 --start 0
  python run_clevr_inference_demo.py --questions_json /path/to/clevr_val_4choices_1k_per_type.json
  python run_clevr_inference_demo.py --option_in_context --no_save --n 4

默认输出: output/clevr/<模型目录名>/inference_demo/（manifest.json、results.json、images/）。
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

from model.load_model import infer, load_model
from project import clevr_v1_0_path, data_path, volcano_7b_luoruipu1_path

# 与 locals.datasets.eval.qa.CLEVRDataset 中 prompt 一致：prompt = "{question}"
def _format_question_text(question: str, options: list[str], option_in_context: bool) -> str:
    if not option_in_context:
        return question
    return question + " Select from following options: " + "; ".join(options) + "."


def _default_questions_json() -> str | None:
    """在 dataset/vo_cot 与 dataset/vo-cot 下寻找 clevr_val_4choices_1k_per_type.json。"""
    for sub in ("vo_cot", "vo-cot"):
        p = os.path.join(data_path, sub, "VoCoT", "eval", "CLEVR", "clevr_val_4choices_1k_per_type.json")
        if os.path.isfile(p):
            return p
    return None


def _default_image_dir() -> str:
    return os.path.join(clevr_v1_0_path, "images", "val")


def _resolve_image_path(image_dir: str, image_filename: str) -> str:
    base = os.path.basename(image_filename)
    p = os.path.join(image_dir, base)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"找不到图像: {p}")
    return p


def _resolve_model_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.path.isdir(volcano_7b_luoruipu1_path):
        return volcano_7b_luoruipu1_path
    return "luoruipu1/Volcano-7b"


def _default_output_dir(model_path: str) -> str:
    ap = os.path.abspath(model_path)
    store = os.path.basename(ap.rstrip(os.sep))
    if not store or store in (".", ".."):
        store = "model"
    return os.path.join(_REPO_ROOT, "output", "clevr", store, "inference_demo")


def main() -> None:
    p = argparse.ArgumentParser(description="CLEVR 推理试跑（4-choice JSON + val 图像）")
    dq = _default_questions_json()
    p.add_argument(
        "--questions_json",
        type=str,
        default=dq,
        help="CLEVRDataset 格式 JSON（list，含 question/answer/answer_options/image_filename）；默认自动探测 vo_cot/VoCoT/eval/CLEVR/clevr_val_4choices_1k_per_type.json",
    )
    p.add_argument("--image_dir", type=str, default=None, help="图像目录，默认 <clevr_v1_0>/images/val")
    p.add_argument(
        "--option_in_context",
        action="store_true",
        help="与 CLEVRDataset.option_in_context 一致：在问题后附加四选项文本",
    )
    p.add_argument("--n", type=int, default=10, help="连续跑多少条（json list 下标顺序）")
    p.add_argument("--start", type=int, default=0, help="起始下标（0-based）")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--precision", type=str, default="fp16", choices=("fp16", "bf16", "fp32"))
    p.add_argument("--no_cot", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--jsonl", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--no_save", action="store_true")
    p.add_argument("--no_copy_images", action="store_true")
    args = p.parse_args()

    if not args.questions_json:
        raise FileNotFoundError(
            "未找到默认 questions JSON。请指定 --questions_json，或把 clevr_val_4choices_1k_per_type.json "
            "放到 dataset/vo_cot/VoCoT/eval/CLEVR/（或 vo-cot 同路径）"
        )
    qpath = os.path.abspath(args.questions_json)
    if not os.path.isfile(qpath):
        raise FileNotFoundError(f"找不到 questions JSON: {qpath}")

    image_dir = args.image_dir or _default_image_dir()
    image_dir = os.path.abspath(image_dir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")

    with open(qpath, encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, list) or not meta:
        raise ValueError("questions JSON 须为非空 list")

    indices = list(range(len(meta)))
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(indices)

    if args.start < 0 or args.start >= len(meta):
        raise IndexError(f"--start={args.start} 越界（共 {len(meta)} 条）")
    end = min(args.start + max(args.n, 0), len(meta))
    if args.start >= end:
        raise ValueError("没有可跑的样本（检查 --n / --start）")
    run_indices = indices[args.start:end]

    model_path = _resolve_model_path(args.model_path)
    out_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else _default_output_dir(model_path)
    )
    if not args.no_save:
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    if not args.jsonl:
        print(f"# model_path: {model_path}", file=sys.stderr)
        print(f"# questions_json: {qpath}", file=sys.stderr)
        print(f"# image_dir: {image_dir}", file=sys.stderr)
        print(f"# 本批条数: {len(run_indices)}", file=sys.stderr)
        if not args.no_save:
            print(f"# 输出目录: {out_dir}", file=sys.stderr)

    model, preprocessor = load_model(model_path, device=args.device, precision=args.precision)

    results_rows: list[dict] = []

    for row_idx in run_indices:
        item = meta[row_idx]
        question = item["question"]
        options = item["answer_options"]
        answer = item.get("answer")
        image_filename = item["image_filename"]
        query_text = _format_question_text(question, options, args.option_in_context)

        img_path = _resolve_image_path(image_dir, image_filename)
        image = Image.open(img_path).convert("RGB")
        out = infer(
            model,
            preprocessor,
            image,
            query_text,
            cot=not args.no_cot,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        text = out[0] if isinstance(out, (list, tuple)) else out

        item_id = f"eval_{row_idx}"
        base_img = os.path.basename(image_filename)
        rel_saved = f"images/{row_idx:04d}_{base_img}"
        if not args.no_save and not args.no_copy_images:
            shutil.copy2(img_path, os.path.join(out_dir, rel_saved))

        rec = {
            "item_id": item_id,
            "index": row_idx,
            "question": question,
            "answer_options": options,
            "answer": answer,
            "image_filename": image_filename,
            "source_image_path": img_path,
            "prediction": text,
            "option_in_context": args.option_in_context,
        }
        if not args.no_save and not args.no_copy_images:
            rec["saved_image"] = rel_saved
        elif not args.no_save:
            rec["saved_image"] = None

        results_rows.append(rec)

        if args.jsonl:
            print(json.dumps(rec, ensure_ascii=False))
        else:
            print("-" * 60)
            print(f"item_id: {item_id}")
            print(f"image:   {img_path}")
            print(f"Q:       {query_text[:200]}{'…' if len(query_text) > 200 else ''}")
            print("-" * 60)
            print(text)
            print()

    if not args.no_save:
        ts = datetime.now(timezone.utc).isoformat()
        manifest = {
            "created_at_utc": ts,
            "output_dir": out_dir,
            "questions_json": qpath,
            "image_dir": image_dir,
            "clevr_v1_0_path": os.path.abspath(clevr_v1_0_path),
            "model_path": model_path,
            "device": args.device,
            "precision": args.precision,
            "no_cot": args.no_cot,
            "option_in_context": args.option_in_context,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "shuffle": args.shuffle,
            "seed": args.seed,
            "start": args.start,
            "n": args.n,
            "run_indices": run_indices,
            "no_copy_images": args.no_copy_images,
            "eval_note": "可选: python eval/eval_tools/clevr.py --result <本目录>/results.json --config <config/datasets/eval/CLEVR_val1k_*.yaml>（需 path/image_dir 与本次一致）",
        }
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results_rows, f, ensure_ascii=False, indent=2)
        if not args.jsonl:
            print("# 已写入 manifest.json、results.json", file=sys.stderr)
            if not args.no_copy_images:
                print(f"# 图片副本: {os.path.join(out_dir, 'images')}", file=sys.stderr)


if __name__ == "__main__":
    main()
