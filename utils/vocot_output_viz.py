"""
解析 VoCoT 输出中的 <coor> 框，并在「扩方图」或「原图」上绘制。
坐标与训练/推理一致：归一化到 [0,1]，相对 expand-to-square 后的图像。
"""
from __future__ import annotations

import json
import re
from typing import List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

# 与 locals.datasets.preprocessor.VoCoT_InputProcessor.expand2square_fn 一致（默认 image_mean 来自 CLIP）
def expand2square(pil_img: Image.Image, background_color: Tuple[int, int, int] = (122, 116, 104)) -> Image.Image:
    width, height = pil_img.size
    if width == height:
        return pil_img.copy()
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result


def extract_coor_boxes_mistral(text: str) -> List[Tuple[float, float, float, float]]:
    """从生成文本中解析所有 <coor> x,y,x,y </coor>（归一化 0–1，相对扩方图）。"""
    pat = re.compile(
        r"<coor>\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*</coor>",
        re.IGNORECASE,
    )
    out: List[Tuple[float, float, float, float]] = []
    for m in pat.finditer(text):
        out.append(tuple(float(m.group(i)) for i in range(1, 5)))
    return out


def norm_box_to_square_pixels(
    box: Tuple[float, float, float, float], square_size: int
) -> Tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box
    S = float(square_size)
    return (
        int(xmin * S),
        int(ymin * S),
        int(xmax * S),
        int(ymax * S),
    )


def square_pixels_to_original(
    box_sq: Tuple[int, int, int, int],
    orig_w: int,
    orig_h: int,
    square_side: int,
) -> Tuple[int, int, int, int]:
    """将扩方图上的像素框映射回原图坐标（裁剪粘贴的逆变换）。"""
    x1, y1, x2, y2 = box_sq
    if orig_w >= orig_h:
        dy = (orig_w - orig_h) // 2
        ox1, oy1, ox2, oy2 = x1, y1 - dy, x2, y2 - dy
    else:
        dx = (orig_h - orig_w) // 2
        ox1, oy1, ox2, oy2 = x1 - dx, y1, x2 - dx, y2

    def clip(v: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, v))

    ox1 = clip(ox1, 0, orig_w - 1)
    ox2 = clip(ox2, 0, orig_w - 1)
    oy1 = clip(oy1, 0, orig_h - 1)
    oy2 = clip(oy2, 0, orig_h - 1)
    if ox2 <= ox1:
        ox2 = min(orig_w - 1, ox1 + 1)
    if oy2 <= oy1:
        oy2 = min(orig_h - 1, oy1 + 1)
    return ox1, oy1, ox2, oy2


def draw_boxes(
    image: Image.Image,
    boxes_px: Sequence[Tuple[int, int, int, int]],
    labels: Sequence[str],
    colors: Sequence[Tuple[int, int, int]] | None = None,
    width: int = 3,
) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    if colors is None:
        palette = [
            (255, 64, 64),
            (64, 128, 255),
            (64, 200, 64),
            (255, 180, 0),
            (200, 64, 200),
            (0, 180, 180),
        ]
        colors = [palette[i % len(palette)] for i in range(len(boxes_px))]
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, (b, lab, c) in enumerate(zip(boxes_px, labels, colors)):
        draw.rectangle([b[0], b[1], b[2], b[3]], outline=c, width=width)
        # 序号标签
        tx, ty = b[0], max(0, b[1] - 2)
        draw.text((tx, ty), str(lab), fill=c, font=font)
    return img
