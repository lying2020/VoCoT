#!/usr/bin/env python3
"""
VoCoT / VolCano 最简推理示例：单图 + 文本 -> 文本（可选 VoCoT 链式推理与坐标）。

用法（在 VoCoT 仓库根目录执行）:
  python run_inference_demo.py
  python run_inference_demo.py --model_path luoruipu1/Volcano-7b --image images/sample_input.jpg --query "Describe the image."

默认模型路径：优先使用 project.py 中的 volcano_7b_luoruipu1_path（若该本地目录存在），否则为 Hugging Face id luoruipu1/Volcano-7b。
默认图像：若存在 test_images/default.jpg 则优先，否则使用 images/sample_input.jpg（见 project.images_dir）。

依赖: 已按 README 安装 requirements.txt 与 flash-attn（可选）。
"""
from __future__ import annotations

import argparse
import os
import sys

# 本脚本位于 VoCoT 仓库根目录
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PIL import Image

from model.load_model import infer, load_model
from project import images_dir, volcano_7b_luoruipu1_path


def _default_image_path() -> str:
    p_custom = os.path.join(images_dir, "default.jpg")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="VoCoT VolCano inference demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="本地模型目录或 Hugging Face id；省略时若 project.volcano_7b_luoruipu1_path 存在则用之，否则 luoruipu1/Volcano-7b",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=_default_image_path(),
        help="输入图像；默认优先 test_images/default.jpg，否则 images/sample_input.jpg",
    )
    parser.add_argument("--query", type=str, default="Describe the image.", help="文本问题")
    parser.add_argument("--no_cot", action="store_true", help="关闭 VoCoT（不附加 grounding/COT 提示）")
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=("fp16", "bf16", "fp32"),
        help="推理精度；无 BF16 的 GPU 请用 fp16",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 或 cpu（不推荐）")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="不打印选用的模型路径等信息",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"找不到图像: {args.image}")

    model_path = _resolve_model_path(args.model_path)
    if not args.quiet:
        print(f"# model_path: {model_path}", file=sys.stderr)
        print(f"# image: {args.image}", file=sys.stderr)

    image = Image.open(args.image).convert("RGB")
    model, preprocessor = load_model(model_path, device=args.device, precision=args.precision)

    out = infer(
        model,
        preprocessor,
        image,
        args.query,
        cot=not args.no_cot,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    # infer 返回 list[str]
    text = out[0] if isinstance(out, (list, tuple)) else out
    print(text)


if __name__ == "__main__":
    main()
