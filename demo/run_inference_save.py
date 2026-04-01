#!/usr/bin/env python3
"""
推理并将文本 + 框可视化保存到 output/ 目录。

  python run_inference_save.py --image images/22.png --query "..." --out_dir output/run1

多轮（每轮独立 infer，无对话历史）：

  python run_inference_save.py --image images/22.png --rounds_json rounds.json --out_dir output/mr

rounds.json: ["Question 1", "Question 2"]

单次多轮对话（一次 generate，infer_multiturn）：

  python run_inference_save.py --image images/22.png --multiturn_dialog examples/multiturn_dialog.json --out_dir output/mt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PIL import Image

from model.load_model import infer, infer_multiturn, load_model
from project import images_dir, volcano_7b_luoruipu1_path
from utils.vocot_output_viz import (
    draw_boxes,
    expand2square,
    extract_coor_boxes_mistral,
    norm_box_to_square_pixels,
    square_pixels_to_original,
)


def _default_image_path() -> str:
    # p_custom = os.path.join(images_dir, "default.jpg")
    p_custom = "/home1/cjl/MM_2026/dataset/GQA_Bench/images/images/2363419.jpg"
    p_figs = os.path.join(_REPO_ROOT, "images", "sample_input.jpg")
    if os.path.isfile(p_custom):
        return p_custom
    return p_figs


def _resolve_model_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.path.isdir(volcano_7b_luoruipu1_path):
        return volcano_7b_luoruipu1_path
    return "luoruipu1/Volcano-7b"


def _load_rounds_queries(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and data and isinstance(data[0], str):
        return data
    raise ValueError("rounds_json 须为 JSON 字符串数组，例如 [\"Q1\", \"Q2\"]")


def _load_multiturn_dialog(path: str) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    turns: list[tuple[str, str]] = []
    for x in data:
        role = str(x.get("role", "")).lower()
        content = str(x.get("content", ""))
        if role in ("user", "human"):
            turns.append(("user", content))
        elif role in ("assistant", "gpt"):
            turns.append(("assistant", content))
        else:
            raise ValueError(f"无效 role: {role}")
    return turns


def _save_run(
    out_dir: str,
    image_path: str,
    pil_orig: Image.Image,
    rounds: list[dict],
    model_sub_image_bind: bool,
):
    os.makedirs(out_dir, exist_ok=True)

    # 说明文档（RefBind / 维度）
    ref_src = os.path.join(_REPO_ROOT, "doc", "REFBIND_AND_DIMENSIONS.md")
    if os.path.isfile(ref_src):
        shutil.copy2(ref_src, os.path.join(out_dir, "REFBIND_AND_DIMENSIONS.md"))

    meta = {
        "image": os.path.abspath(image_path),
        "sub_image_bind": model_sub_image_bind,
        "refbind_used": model_sub_image_bind,
        "note": "refbind_used=True 表示子图裁剪+encode（generate_sub_image）；False 表示整图 box_align（generate_box）。",
        "rounds": rounds,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    sq = expand2square(pil_orig)
    S = sq.size[0]
    ow, oh = pil_orig.size

    all_boxes_order: list[tuple[int, int, int, int]] = []
    all_labels: list[str] = []

    global_idx = 1
    for ri, r in enumerate(rounds):
        text = r["text"]
        tag = r.get("tag", f"round_{ri+1:02d}")
        with open(os.path.join(out_dir, f"{tag}_text.txt"), "w", encoding="utf-8") as f:
            f.write(text)

        boxes = extract_coor_boxes_mistral(text)
        r["num_boxes"] = len(boxes)

        boxes_sq_px = [norm_box_to_square_pixels(b, S) for b in boxes]
        labels_round = [str(global_idx + i) for i in range(len(boxes_sq_px))]
        global_idx += len(boxes_sq_px)

        img_sq = draw_boxes(sq, boxes_sq_px, labels_round)
        img_sq.save(os.path.join(out_dir, f"{tag}_viz_square.png"))

        boxes_orig = [square_pixels_to_original(b, ow, oh, S) for b in boxes_sq_px]
        img_og = draw_boxes(pil_orig, boxes_orig, labels_round)
        img_og.save(os.path.join(out_dir, f"{tag}_viz_original.png"))

        all_boxes_order.extend(boxes_orig)
        all_labels.extend(labels_round)

    if all_boxes_order:
        img_all = draw_boxes(pil_orig, all_boxes_order, all_labels)
        img_all.save(os.path.join(out_dir, "combined_all_rounds_viz_original.png"))

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"sub_image_bind (RefBind path) = {model_sub_image_bind}\n")
        f.write(f"Total boxes across rounds: {len(all_boxes_order)}\n\n")
        for r in rounds:
            f.write(f"--- {r.get('tag')} ---\n{r['text']}\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="VoCoT inference + save viz to output/")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--image", type=str, default=_default_image_path())
    # parser.add_argument("--query", type=str, default="Describe the image.")
    parser.add_argument("--query", type=str, default="Do all these people have the same gender?", help="文本问题")
    parser.add_argument("--no_cot", action="store_true")
    parser.add_argument("--precision", type=str, default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="输出目录；默认 output/run_时间戳",
    )
    parser.add_argument(
        "--rounds_json",
        type=str,
        default=None,
        help="多轮（多次独立 infer）问题列表 JSON 数组",
    )
    parser.add_argument(
        "--multiturn_dialog",
        type=str,
        default=None,
        help="单次 infer_multiturn 的对话 JSON（格式同 examples/multiturn_dialog.json）",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(args.image)

    out_dir = args.out_dir or os.path.join(_REPO_ROOT, "output", f"run_{time.strftime('%Y%m%d_%H%M%S')}")

    model_path = _resolve_model_path(args.model_path)
    print(f"# model_path: {model_path}", file=sys.stderr)
    print(f"# out_dir: {out_dir}", file=sys.stderr)

    pil_orig = Image.open(args.image).convert("RGB")
    model, preprocessor = load_model(model_path, device=args.device, precision=args.precision)
    sub_bind = bool(getattr(model, "sub_image_bind", False))

    rounds: list[dict] = []

    if args.multiturn_dialog:
        turns = _load_multiturn_dialog(args.multiturn_dialog)
        out = infer_multiturn(
            model,
            preprocessor,
            pil_orig,
            turns,
            cot_last=not args.no_cot,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        text = out[0] if isinstance(out, (list, tuple)) else out
        rounds.append({"tag": "multiturn_single_reply", "text": text})
        print(text)
    elif args.rounds_json:
        queries = _load_rounds_queries(args.rounds_json)
        for i, q in enumerate(queries):
            out = infer(
                model,
                preprocessor,
                pil_orig,
                q,
                cot=not args.no_cot,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            text = out[0] if isinstance(out, (list, tuple)) else out
            rounds.append({"tag": f"round_{i+1:02d}", "text": text})
            print(f"=== round {i+1} ===\n{text}\n", file=sys.stderr)
        print(rounds[-1]["text"] if rounds else "")
    else:
        out = infer(
            model,
            preprocessor,
            pil_orig,
            args.query,
            cot=not args.no_cot,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        text = out[0] if isinstance(out, (list, tuple)) else out
        rounds.append({"tag": "single", "text": text})
        print(text)

    _save_run(out_dir, args.image, pil_orig, rounds, sub_bind)
    print(f"# saved to: {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
