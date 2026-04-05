#!/usr/bin/env python3
"""
单张图 + 自定义问题，本地推理一条（不依赖 GQA JSON）。

默认图片: examples/img_v3_gqa_dataset.jpg（相对 VoCoT 仓库根）
默认问题: 脚本内建多步推理提示（可用 --question / --question_file 覆盖）

在仓库根或任意目录执行均可:
  python demo/run_gqa_single_case.py
  python demo/run_gqa_single_case.py --image /path/to.jpg --question "What is ..."
  python demo/run_gqa_single_case.py --question_file prompts.txt

结果默认写入: output/gqa_single/<模型目录名>/single_case/
（原图副本 images/<stem>.jpg；若预测中含 <coor>，另存映射回原图的框选图 images/<stem>_with_boxes.jpg）
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def _find_vocot_root() -> str:
    here = Path(__file__).resolve()
    for d in [here.parent, *here.parents]:
        if (d / "model" / "load_model.py").is_file():
            return str(d)
    raise RuntimeError(
        f"无法定位 VoCoT 仓库根（需存在 model/load_model.py）。当前脚本: {here}"
    )


_REPO_ROOT = _find_vocot_root()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.load_model import infer, load_model
from project import volcano_7b_luoruipu1_path
from utils.vocot_output_viz import (
    draw_boxes,
    expand2square,
    extract_coor_boxes_mistral,
    norm_box_to_square_pixels,
    square_pixels_to_original,
)

# 相对仓库根的默认相对路径（与 MM_2026/VoCoT/examples/... 一致）
DEFAULT_IMAGE_REL = os.path.join("examples", "img_v3_gqa_dataset.jpg")

# DEFAULT_QUESTION = """ Based on the scene in the image and the behavior of the other people, what is the woman in the yellow jacket most likely about to do next? """

# DEFAULT_QUESTION = """Question: What is the woman in the yellow jacket on the left about to do? Please follow five steps to arrive at the answer.

# First, locate the woman in the yellow jacket.

# Then, observe the space she is in.

# Next, look at the group of people on the right side.

# Then, observe what they are carrying.

# Finally, derive the answer."""

DEFAULT_QUESTION = """Question:  Locate the woman in yellow jacket, and find the place she is waiting for, and find a queue of people on the right standing in a line, and find the luggage and bags can be spotted."""

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
    return os.path.join(_REPO_ROOT, "output", "gqa_single", store, "single_case")


def _resolve_image_path(user_path: str | None) -> str:
    if not user_path:
        p = os.path.join(_REPO_ROOT, DEFAULT_IMAGE_REL)
    else:
        p = os.path.abspath(os.path.expanduser(user_path))
        if not os.path.isabs(user_path) and not os.path.isfile(p):
            alt = os.path.join(_REPO_ROOT, user_path)
            if os.path.isfile(alt):
                p = alt
    if not os.path.isfile(p):
        raise FileNotFoundError(f"找不到图像: {p}")
    return p


def _resolve_question(args: argparse.Namespace) -> str:
    if args.question_file:
        with open(args.question_file, encoding="utf-8") as f:
            q = f.read()
    elif args.question is not None:
        q = args.question
    else:
        q = DEFAULT_QUESTION
    q = q.strip()
    if not q:
        raise ValueError("问题文本为空")
    return q


def main() -> None:
    p = argparse.ArgumentParser(description="GQA 风格单图单题本地推理")
    p.add_argument(
        "--image",
        type=str,
        default=None,
        help=f"图片路径；默认仓库内 {DEFAULT_IMAGE_REL}",
    )
    p.add_argument(
        "--question",
        type=str,
        default=None,
        help="问题全文；不填则使用脚本默认多步提示",
    )
    p.add_argument(
        "--question_file",
        type=str,
        default=None,
        help="从文件读取问题（UTF-8），优先级高于 --question",
    )
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--precision", type=str, default="fp16", choices=("fp16", "bf16", "fp32"))
    p.add_argument("--no_cot", action="store_true", help="关闭 VoCoT/grounding+CoT 提示")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--jsonl", action="store_true", help="仅一行 JSON 输出预测与元数据")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--no_save", action="store_true")
    p.add_argument("--no_copy_images", action="store_true")
    args = p.parse_args()

    img_path = _resolve_image_path(args.image)
    question = _resolve_question(args)

    model_path = _resolve_model_path(args.model_path)
    out_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else _default_output_dir(model_path)
    )
    stem = Path(img_path).stem
    rel_saved = f"images/{stem}.jpg"
    rel_saved_boxes = f"images/{stem}_with_boxes.jpg"

    if not args.jsonl:
        print(f"# model_path: {model_path}", file=sys.stderr)
        print(f"# image: {img_path}", file=sys.stderr)
        print(f"# output_dir: {out_dir if not args.no_save else '(no_save)'}", file=sys.stderr)

    if not args.no_save:
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    model, preprocessor = load_model(model_path, device=args.device, precision=args.precision)
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

    if not args.no_save and not args.no_copy_images:
        dst_img = os.path.join(out_dir, rel_saved)
        shutil.copy2(img_path, dst_img)

    # 解析 <coor>（相对扩方图 [0,1]），映射回原图画框；与 run_inference_save / 训练一致
    coor_boxes = extract_coor_boxes_mistral(text)
    saved_boxes_rel: str | None = None
    if (
        coor_boxes
        and not args.no_save
        and not args.no_copy_images
    ):
        pil_orig = image
        ow, oh = pil_orig.size
        sq = expand2square(pil_orig)
        S = int(sq.size[0])
        boxes_sq_px = [norm_box_to_square_pixels(b, S) for b in coor_boxes]
        labels = [str(i + 1) for i in range(len(boxes_sq_px))]
        boxes_orig = [
            square_pixels_to_original(b, ow, oh, S) for b in boxes_sq_px
        ]
        lw = max(2, int(round(min(ow, oh) * 0.004)))
        img_boxed = draw_boxes(pil_orig, boxes_orig, labels, width=lw)
        dst_box = os.path.join(out_dir, rel_saved_boxes)
        img_boxed.save(dst_box, format="JPEG", quality=95)
        saved_boxes_rel = rel_saved_boxes
        if not args.jsonl:
            print(
                f"# 已从输出解析 {len(coor_boxes)} 个 <coor>，已保存框图: {dst_box}",
                file=sys.stderr,
            )
    elif coor_boxes and not args.jsonl:
        print(
            f"# 已从输出解析 {len(coor_boxes)} 个 <coor>（未保存框图：--no_save 或 --no_copy_images）",
            file=sys.stderr,
        )

    rec = {
        "source_image_path": img_path,
        "question": question,
        "prediction": text,
        "num_coor_boxes": len(coor_boxes),
    }
    if not args.no_save:
        rec["saved_image"] = None if args.no_copy_images else rel_saved
        rec["saved_image_with_boxes"] = (
            None if args.no_copy_images else saved_boxes_rel
        )

    if args.jsonl:
        print(json.dumps(rec, ensure_ascii=False))
    else:
        print("-" * 60)
        print(f"image: {img_path}")
        print(f"question:\n{question}")
        print("-" * 60)
        print(text)
        print()

    if not args.no_save:
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": out_dir,
            "model_path": model_path,
            "device": args.device,
            "precision": args.precision,
            "no_cot": args.no_cot,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "image": img_path,
            "question_source": (
                "file:" + os.path.abspath(args.question_file)
                if args.question_file
                else ("arg:--question" if args.question is not None else "builtin_default")
            ),
            "no_copy_images": args.no_copy_images,
            "num_coor_boxes": len(coor_boxes),
            "saved_image_with_boxes": saved_boxes_rel,
        }
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump([rec], f, ensure_ascii=False, indent=2)
        if not args.jsonl:
            print("# 已写入 manifest.json、results.json", file=sys.stderr)
            if not args.no_copy_images:
                print(f"# 原图副本: {os.path.join(out_dir, rel_saved)}", file=sys.stderr)
                if saved_boxes_rel:
                    print(
                        f"# 框选图: {os.path.join(out_dir, saved_boxes_rel)}",
                        file=sys.stderr,
                    )


if __name__ == "__main__":
    main()
