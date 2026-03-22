#!/usr/bin/env python3
"""
多轮对话 + 多 HOP 文档式推理 调测脚本。

两种模式：
  1) document — 单次生成，要求模型按 ### HOP 1/2/... 输出（见 constants.MULTIHOP_DOC_INSTRUCTION）
  2) multiturn — 同一张图上的多轮 user/assistant 历史 + 最后一问，一次生成当前轮回复

示例：
  python run_multihop_demo.py --mode document --query "分析图中场景并做多跳推理。"
  python run_multihop_demo.py --mode multiturn --dialog examples/multiturn_dialog.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PIL import Image

from model.load_model import infer_multihop_document, infer_multiturn, load_model
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


def _load_dialog_turns(path: str) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON 须为 list，每项含 role 与 content")
    turns: list[tuple[str, str]] = []
    for i, x in enumerate(data):
        role = str(x.get("role", "")).lower()
        content = str(x.get("content", ""))
        if role in ("user", "human"):
            turns.append(("user", content))
        elif role in ("assistant", "gpt"):
            turns.append(("assistant", content))
        else:
            raise ValueError(f"第 {i} 条 role 无效: {role}")
    return turns


def main() -> None:
    parser = argparse.ArgumentParser(description="VoCoT multi-hop / multi-turn demo")
    parser.add_argument(
        "--mode",
        type=str,
        choices=("document", "multiturn"),
        default="document",
        help="document=单次多 HOP 文档；multiturn=多轮对话（JSON）",
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--image", type=str, default=_default_image_path())
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", type=str, default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no_cot", action="store_true", help="关闭最后一轮 VoCoT COT（仅 multiturn 末轮或 document 的 infer）")

    # document
    parser.add_argument("--query", type=str, default="", help="[document] 用户问题")
    parser.add_argument(
        "--hop_instruction_file",
        type=str,
        default=None,
        help="[document] 覆盖默认 MULTIHOP_DOC_INSTRUCTION 的文本文件",
    )

    # multiturn
    parser.add_argument(
        "--dialog",
        type=str,
        default=None,
        help="[multiturn] JSON 路径，格式见 examples/multiturn_dialog.json",
    )
    parser.add_argument(
        "--print_turns",
        action="store_true",
        help="[multiturn] 在 stderr 打印解析后的轮次（调试用）",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"找不到图像: {args.image}")

    model_path = _resolve_model_path(args.model_path)
    print(f"# model_path: {model_path}", file=sys.stderr)
    print(f"# image: {args.image}", file=sys.stderr)
    print(f"# mode: {args.mode}", file=sys.stderr)

    image = Image.open(args.image).convert("RGB")
    model, preprocessor = load_model(model_path, device=args.device, precision=args.precision)

    hop_instruction = None
    if args.hop_instruction_file:
        with open(args.hop_instruction_file, "r", encoding="utf-8") as f:
            hop_instruction = f.read()

    if args.mode == "document":
        query = args.query or "Describe the scene and perform multi-hop reasoning."
        out = infer_multihop_document(
            model,
            preprocessor,
            image,
            query,
            cot=not args.no_cot,
            hop_instruction=hop_instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        if not args.dialog:
            raise SystemExit("multiturn 模式需要 --dialog path/to.json")
        turns = _load_dialog_turns(args.dialog)
        if args.print_turns:
            for i, (r, t) in enumerate(turns):
                preview = t[:120] + ("..." if len(t) > 120 else "")
                print(f"# turn {i} {r}: {preview!r}", file=sys.stderr)
        out = infer_multiturn(
            model,
            preprocessor,
            image,
            turns,
            cot_last=not args.no_cot,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    text = out[0] if isinstance(out, (list, tuple)) else out
    print(text)


if __name__ == "__main__":
    main()
