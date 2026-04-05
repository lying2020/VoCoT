#!/usr/bin/env python3
"""
将含 <coor> 的片段画到指定「原图」上，验证归一化坐标 → expand2square → 原图像素 的映射是否与 VoCoT 一致。

默认使用你给的示例正文 + 默认单 case 输出目录里的原图副本。

  python demo/verify_coor_on_image.py
  python demo/verify_coor_on_image.py --image /path/to.jpg --out /path/to/out.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _find_vocot_root() -> str:
    here = Path(__file__).resolve()
    for d in [here.parent, *here.parents]:
        if (d / "model" / "load_model.py").is_file():
            return str(d)
    raise RuntimeError(f"找不到 VoCoT 根目录（需含 model/load_model.py）。脚本: {here}")


_REPO_ROOT = _find_vocot_root()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PIL import Image

from utils.vocot_output_viz import (
    draw_boxes,
    expand2square,
    extract_coor_boxes_mistral,
    norm_box_to_square_pixels,
    square_pixels_to_original,
)

# 待验证的正文（与模型输出格式一致：<coor> x1,y1,x2,y2 </coor>，相对扩方图 [0,1]）
DEFAULT_SAMPLE_TEXT = r"""Locate the woman in yellow t <coor> 0.100,0.544,0.280,0.662</coor>, who is sitting and resting on a lounge chair  <coor> 0.000, 0.560, 0.280, 0.872</coor>. Near the chair, there is a queue of people on the right   <coor> 0.650, 0.320, 0.860, 0.560 </coor> , standing in a line. Next to the queue, luggage and bags <coor> 0.600, 0.480, 0.680, 0.597 </coor>  can be spotted. Therefore, the woman in yellow is waiting in the terminal and is about to join the queue and board her flight."""

DEFAULT_IMAGE_REL = os.path.join(
    "output",
    "gqa_single",
    "volcano_7b_luoruipu1",
    "single_case",
    "images",
    "img_v3_gqa_dataset.jpg",
)


def main() -> None:
    p = argparse.ArgumentParser(description="在「原图」上绘制 <coor> 框以验证映射")
    p.add_argument(
        "--image",
        type=str,
        default=None,
        help=f"原图路径；默认 {_REPO_ROOT}/{DEFAULT_IMAGE_REL}",
    )
    p.add_argument(
        "--text",
        type=str,
        default=None,
        help="含 <coor> 的整段文字；不填则用脚本内建示例",
    )
    p.add_argument(
        "--text_file",
        type=str,
        default=None,
        help="从 UTF-8 文件读取正文（优先于 --text）",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="框图保存路径；默认 <image 同目录>/basename_coor_verify.jpg",
    )
    p.add_argument(
        "--save_square",
        action="store_true",
        help="额外保存扩方图上的框选图 *_coor_verify_square.jpg",
    )
    p.add_argument(
        "--dump_json",
        type=str,
        default=None,
        help="将解析到的归一化框与映射后原图像素框写入该路径（JSON）",
    )
    args = p.parse_args()

    img_path = args.image or os.path.join(_REPO_ROOT, DEFAULT_IMAGE_REL)
    img_path = os.path.abspath(os.path.expanduser(img_path))
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"找不到图像: {img_path}")

    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            text = f.read()
    elif args.text is not None:
        text = args.text
    else:
        text = DEFAULT_SAMPLE_TEXT

    boxes_norm = extract_coor_boxes_mistral(text)
    if not boxes_norm:
        raise RuntimeError("未解析到任何 <coor>，检查正文或正则是否与输出格式一致")

    pil_orig = Image.open(img_path).convert("RGB")
    ow, oh = pil_orig.size
    sq = expand2square(pil_orig)
    S = int(sq.size[0])

    boxes_sq_px = [norm_box_to_square_pixels(b, S) for b in boxes_norm]
    boxes_orig = [
        square_pixels_to_original(b, ow, oh, S) for b in boxes_sq_px
    ]
    labels = [str(i + 1) for i in range(len(boxes_orig))]
    lw = max(2, int(round(min(ow, oh) * 0.004)))
    img_viz = draw_boxes(pil_orig, boxes_orig, labels, width=lw)

    out_path = args.out
    if not out_path:
        stem = Path(img_path).stem
        out_path = os.path.join(
            os.path.dirname(img_path), f"{stem}_coor_verify.jpg"
        )
    img_viz.save(out_path, quality=95)

    print(f"# 解析到 {len(boxes_norm)} 个框", file=sys.stderr)
    print(f"# 原图: {img_path}  ({ow}x{oh})", file=sys.stderr)
    print(f"# 扩方边长 S={S}", file=sys.stderr)
    for i, (n, sqp, og) in enumerate(
        zip(boxes_norm, boxes_sq_px, boxes_orig), start=1
    ):
        print(
            f"#  {i}  norm={n}  square_px={sqp}  orig_px={og}",
            file=sys.stderr,
        )
    print(f"# 已保存原图框选: {out_path}", file=sys.stderr)

    if args.save_square:
        labels_sq = [str(i + 1) for i in range(len(boxes_sq_px))]
        lw_sq = max(2, int(round(S * 0.004)))
        img_sq_viz = draw_boxes(sq, boxes_sq_px, labels_sq, width=lw_sq)
        sq_out = os.path.join(
            os.path.dirname(out_path),
            f"{Path(out_path).stem}_square{Path(out_path).suffix}",
        )
        img_sq_viz.save(sq_out, quality=95)
        print(f"# 已保存扩方图框选: {sq_out}", file=sys.stderr)

    if args.dump_json:
        payload = {
            "image": img_path,
            "orig_size": [ow, oh],
            "square_side": S,
            "boxes_norm_square_space": [list(map(float, b)) for b in boxes_norm],
            "boxes_pixel_square": [list(map(int, b)) for b in boxes_sq_px],
            "boxes_pixel_original": [list(map(int, b)) for b in boxes_orig],
        }
        with open(args.dump_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"# 已写入 JSON: {args.dump_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
