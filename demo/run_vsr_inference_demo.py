#!/usr/bin/env python3
"""
VSR 数据集本地推理试跑（不跑 evaluate_benchmark 全量评测）。

数据默认来自 project.vsr_ramdom_path 下 train/dev/test.jsonl，图片根目录为 project.vsr_images_path。
若图片按 COCO 子目录存放则为 <base>/<train2017|val2017>/<文件名>；若整包扁平放在 images/ 下，脚本会自动回退为 <base>/<文件名>。

在 VoCoT 仓库根目录执行:
  python run_vsr_inference_demo.py
  python run_vsr_inference_demo.py --split test --n 20 --start 0
  python run_vsr_inference_demo.py --no_save --n 3

保存目录默认: output/vsr/<模型目录名>/inference_demo/（manifest.json、results.json、images/）。
results 中含 item_id / prediction / label，可用 eval/eval_tools/vsr.py 配合对应 config 算准确率。
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
from project import volcano_7b_luoruipu1_path, vsr_images_path, vsr_ramdom_path

# 与 locals.datasets.eval.short_qa.VSRDataset 一致
VSR_EVENT_PROMPT = 'Is there an event "{}" taking place in the image?'


def _jsonl_path(split: str) -> str:
    name = f"{split}.jsonl"
    p = os.path.join(vsr_ramdom_path, name)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"未找到 VSR jsonl: {p}（检查 vsr_ramdom_path）")
    return p


def _load_meta(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_vsr_image_path(base_path: str, item: dict) -> str:
    """
    与 VSRDataset 一致：base_path / split / image；若不存在则尝试 base_path / image（扁平）。
    """
    image = item["image"]
    link = item.get("image_link") or ""
    parts = link.split("/")
    split = parts[-2] if len(parts) >= 2 else ""
    p_split = os.path.join(base_path, split, image) if split else ""
    if p_split and os.path.isfile(p_split):
        return p_split
    p_flat = os.path.join(base_path, image)
    if os.path.isfile(p_flat):
        return p_flat
    raise FileNotFoundError(
        f"找不到图像（已尝试 {p_split!r} 与 {p_flat!r}）。请确认图片已解压到 {base_path}"
    )


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
    return os.path.join(_REPO_ROOT, "output", "vsr", store, "inference_demo")


def main() -> None:
    p = argparse.ArgumentParser(description="VSR 推理试跑（jsonl + 本地图）")
    p.add_argument("--vsr_root", type=str, default=vsr_ramdom_path, help="含 train/dev/test.jsonl 的目录（默认 project.vsr_ramdom_path）")
    p.add_argument(
        "--image_root",
        type=str,
        default=vsr_images_path,
        help="图片根目录（默认 project.vsr_images_path）",
    )
    p.add_argument("--split", type=str, default="test", choices=("train", "dev", "test"))
    p.add_argument("--n", type=int, default=10, help="连续跑多少条（与 jsonl 行顺序一致）")
    p.add_argument("--start", type=int, default=0, help="起始行下标（0-based）")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--precision", type=str, default="fp16", choices=("fp16", "bf16", "fp32"))
    p.add_argument("--no_cot", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--jsonl", action="store_true", help="每行一条 JSON 输出到 stdout")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--no_save", action="store_true", help="不写 manifest/results/图片副本")
    p.add_argument("--no_copy_images", action="store_true", help="只写 JSON，不复制图片")
    args = p.parse_args()

    jsonl_path = os.path.join(os.path.abspath(args.vsr_root), f"{args.split}.jsonl")
    if not os.path.isfile(jsonl_path):
        alt = _jsonl_path(args.split)
        if os.path.isfile(alt):
            jsonl_path = alt
        else:
            raise FileNotFoundError(
                f"找不到 {args.split}.jsonl：已尝试 {jsonl_path} 与 {alt}"
            )

    meta = _load_meta(jsonl_path)
    if not meta:
        raise RuntimeError(f"空数据: {jsonl_path}")

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
        print(f"# jsonl: {jsonl_path}", file=sys.stderr)
        print(f"# image_root: {args.image_root}", file=sys.stderr)
        print(f"# 本批条数: {len(run_indices)}", file=sys.stderr)
        if not args.no_save:
            print(f"# 输出目录: {out_dir}", file=sys.stderr)

    model, preprocessor = load_model(model_path, device=args.device, precision=args.precision)

    results_rows: list[dict] = []

    for local_i, row_idx in enumerate(run_indices):
        item = meta[row_idx]
        caption = item["caption"]
        query_text = VSR_EVENT_PROMPT.format(caption.lower())
        img_path = _resolve_vsr_image_path(args.image_root, item)
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
        rel_saved = f"images/{row_idx:04d}_{item['image']}"
        dst_img = os.path.join(out_dir, rel_saved) if not args.no_save else None
        if not args.no_save and not args.no_copy_images:
            shutil.copy2(img_path, dst_img)

        rec = {
            "item_id": item_id,
            "index": row_idx,
            "caption": caption,
            "relation": item.get("relation"),
            "label": item.get("label"),
            "source_image_path": img_path,
            "prediction": text,
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
            print(f"item_id: {item_id}  index: {row_idx}")
            print(f"image:   {img_path}")
            print(f"caption: {caption}")
            print(f"query:   {query_text}")
            print("-" * 60)
            print(text)
            print()

    if not args.no_save:
        ts = datetime.now(timezone.utc).isoformat()
        manifest = {
            "created_at_utc": ts,
            "output_dir": out_dir,
            "jsonl": os.path.abspath(jsonl_path),
            "image_root": os.path.abspath(args.image_root),
            "split": args.split,
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
            "run_indices": run_indices,
            "no_copy_images": args.no_copy_images,
            "vsr_eval_note": "可用 eval/eval_tools/vsr.py：--data 指向本目录 results.json，--config 中 VSRDataset.path 与上面 jsonl 一致、base_path 与 image_root 一致",
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
