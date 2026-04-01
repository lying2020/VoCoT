#!/usr/bin/env python3
"""
EmbSpatial 数据集本地推理试跑（demo 校验，不跑 evaluate_benchmark 全量评测）。

数据默认来自 project.embspatial_path 下的 embspatial_bench.json / embspatial_sft.json。
Hugging Face 发布的 JSON 中 image 多为 **base64 JPEG**（以 /9j/ 开头）；若为相对路径文件名，
也会在 --image_dir 下查找（与 EmbSpatialDataset 一致）。

可选：对每一步前向（prefill + 逐 token decode）导出各层 self-attn KV cache 新增片段。
默认布局（hierarchical）：<kv_traces>/<样本>/layer_{L}/kv_{H}/trajectory_{T}/token_{n}/kv_pos_modal.npz，
其中 k、v 各 (head_dim,)、pos (1,)、modal (1,)。平面布局（flat）见 utils.kv_trace.save_sample_kv。

在 VoCoT 仓库根目录执行（推荐）:
  python demo/run_embspatial_inference_demo.py
  # 或旧路径: python run_embspatial_inference_demo.py
  python run_embspatial_inference_demo.py --subset sft --n 5
  python run_embspatial_inference_demo.py --image_dir /path/to/images --n 2
  python run_embspatial_inference_demo.py --no_save --option_in_context
  python run_embspatial_inference_demo.py --no_kv_trace
  python run_embspatial_inference_demo.py --kv_out_dir /data/emb_kv

默认输出: output/embspatial/<模型目录名>/inference_demo/
默认样本数: --n 100
"""
from __future__ import annotations
 
import argparse
import base64
import binascii
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path


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

from PIL import Image

from model.load_model import infer, load_model
from project import embspatial_path, volcano_7b_luoruipu1_path
from utils.kv_trace import (
    MistralKVTracer,
    patch_volcano_generators_for_modal,
    save_sample_kv,
    save_sample_kv_hierarchical,
)


def _format_question(
    question: str, options: list[str] | None, option_in_context: bool
) -> str:
    if not option_in_context or not options:
        return question
    return question + " Select from following options: " + "; ".join(options) + "."


def _resolve_image_pil(
    raw: str, image_dir: str | None
) -> tuple[Image.Image, str | None]:
    """
    返回 (RGB PIL 图, 若成功从磁盘读则返回绝对路径，否则 None 表示来自 base64)。
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("空 image 字段")

    if os.path.isfile(raw):
        return Image.open(raw).convert("RGB"), os.path.abspath(raw)

    if image_dir:
        p = os.path.join(image_dir, os.path.basename(raw))
        if os.path.isfile(p):
            return Image.open(p).convert("RGB"), os.path.abspath(p)

    try:
        data = base64.b64decode(raw, validate=False)
        im = Image.open(BytesIO(data)).convert("RGB")
        return im, None
    except (binascii.Error, OSError, ValueError):
        pass

    raise FileNotFoundError(
        f"无法加载图像：既非有效 base64 JPEG，也不在磁盘上。"
        f"（可设置 --image_dir；当前 image_dir={image_dir!r}）"
    )


def _default_questions_path(subset: str) -> str:
    name = "embspatial_bench.json" if subset == "bench" else "embspatial_sft.json"
    return os.path.join(embspatial_path, name)


def _load_meta(path: str, subset: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, list) or not meta:
        raise ValueError(f"{path} 须为非空 JSON list")
    if subset == "bench":
        # 与 locals.datasets.eval.qa.EmbSpatialDataset.__init__ 一致
        meta = [x for x in meta if x.get("relation") not in ("on top of", "inside")]
    return meta


def _get_question_and_options(item: dict, subset: str, sft_q_index: int) -> tuple[str, list[str] | None]:
    if subset == "bench":
        return item["question"], list(item.get("answer_options") or [])
    # sft
    qs = item.get("questions") or []
    if not qs:
        raise KeyError("SFT 条目缺少 questions 列表")
    idx = max(0, min(sft_q_index, len(qs) - 1))
    return qs[idx], None


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
    return os.path.join(_REPO_ROOT, "output", "embspatial", store, "inference_demo")


def main() -> None:
    p = argparse.ArgumentParser(description="EmbSpatial 推理 demo（bench / sft）")
    p.add_argument("--emb_root", type=str, default=embspatial_path, help="embspatial 根目录")
    p.add_argument(
        "--subset",
        type=str,
        default="bench",
        choices=("bench", "sft"),
        help="使用 embspatial_bench.json 或 embspatial_sft.json",
    )
    p.add_argument(
        "--questions_json",
        type=str,
        default=None,
        help="覆盖默认的 embspatial_*.json 路径",
    )
    p.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="若 image 为文件名则在此目录查找（可为空，仅 base64 时不需要）",
    )
    p.add_argument("--option_in_context", action="store_true")
    p.add_argument(
        "--sft_question_idx",
        type=int,
        default=0,
        help="sft 条目中 questions 列表使用的下标",
    )
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--start", type=int, default=0)
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
    p.add_argument(
        "--no_kv_trace",
        action="store_true",
        help="关闭各层 KV 导出（更快、省盘）",
    )
    p.add_argument(
        "--kv_out_dir",
        type=str,
        default=None,
        help="KV 数据根目录；默认 <output_dir>/kv_traces",
    )
    p.add_argument(
        "--kv_dtype",
        type=str,
        default="float16",
        choices=("float16", "float32"),
        help="KV 中 k、v 的 numpy dtype（hierarchical 单文件与 flat 的 k_layers 均使用）",
    )
    p.add_argument(
        "--kv_layout",
        type=str,
        default="hierarchical",
        choices=("hierarchical", "flat"),
        help="hierarchical: layer_*/kv_*/trajectory_*/token_* 下 kv_pos_modal.npz；flat: steps/step_*.npz",
    )
    p.add_argument(
        "--kv_trajectory_idx",
        type=int,
        default=0,
        help="轨迹编号（写入 trajectory_{idx}；同一样本多次采样可递增）",
    )
    args = p.parse_args()

    qpath = os.path.abspath(
        args.questions_json
        if args.questions_json
        else _default_questions_path(args.subset)
    )
    if not os.path.isfile(qpath):
        raise FileNotFoundError(f"找不到数据 JSON: {qpath}")

    image_dir = args.image_dir
    if image_dir:
        image_dir = os.path.abspath(image_dir)

    meta = _load_meta(qpath, args.subset)

    indices = list(range(len(meta)))
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(indices)

    if args.start < 0 or args.start >= len(meta):
        raise IndexError(f"--start={args.start} 越界（共 {len(meta)} 条）")
    end = min(args.start + max(args.n, 0), len(meta))
    if args.start >= end:
        raise ValueError("没有可跑样本（检查 --n / --start）")
    run_indices = indices[args.start:end]

    model_path = _resolve_model_path(args.model_path)
    out_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else _default_output_dir(model_path)
    )
    if not args.no_save:
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    kv_tracer: MistralKVTracer | None = None
    kv_root: str | None = None
    if not args.no_kv_trace:
        kv_root = (
            os.path.abspath(args.kv_out_dir)
            if args.kv_out_dir
            else os.path.join(out_dir, "kv_traces")
        )
        os.makedirs(kv_root, exist_ok=True)

    if not args.jsonl:
        print(f"# model_path: {model_path}", file=sys.stderr)
        print(f"# questions_json: {qpath}", file=sys.stderr)
        print(f"# subset: {args.subset} 本批: {len(run_indices)}", file=sys.stderr)
        if image_dir:
            print(f"# image_dir: {image_dir}", file=sys.stderr)
        if not args.no_save:
            print(f"# 输出目录: {out_dir}", file=sys.stderr)
        if kv_root:
            print(
                f"# KV 导出目录: {kv_root}  layout={args.kv_layout}",
                file=sys.stderr,
            )

    model, preprocessor = load_model(model_path, device=args.device, precision=args.precision)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (
        int(cfg.hidden_size) // int(cfg.num_attention_heads)
    )
    kv_model_meta = {
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_key_value_heads": int(cfg.num_key_value_heads),
        "head_dim": int(head_dim),
        "hidden_size": int(cfg.hidden_size),
        "n_query": int(getattr(model, "n_query", getattr(cfg, "num_query_token", 0)) or 0),
    }

    if not args.no_kv_trace:
        kv_tracer = MistralKVTracer(enabled=True)
        kv_tracer.register(model)
        patch_volcano_generators_for_modal(model, kv_tracer)

    results_rows: list[dict] = []

    for row_idx in run_indices:
        item = meta[row_idx]
        qtext, options = _get_question_and_options(
            item, args.subset, args.sft_question_idx
        )
        query_text = _format_question(qtext, options, args.option_in_context)

        qid = item.get("question_id", f"row_{row_idx}")
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(qid))[
            :120
        ]

        image, src_path = _resolve_image_pil(item["image"], image_dir)
        kv_sample_meta = {
            **kv_model_meta,
            "row_idx": int(row_idx),
            "question_id": item.get("question_id"),
            "subset": args.subset,
            "questions_json": qpath,
        }
        out = infer(
            model,
            preprocessor,
            image,
            query_text,
            cot=not args.no_cot,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            kv_tracer=kv_tracer,
            kv_sample_meta=kv_sample_meta,
        )
        sample_kv_dir: str | None = None
        if kv_tracer is not None:
            pred, _imgs, _ids = out
            assert kv_root is not None
            sample_kv_dir = os.path.join(kv_root, f"{row_idx:04d}_{safe_id}")
            if args.kv_layout == "hierarchical":
                save_sample_kv_hierarchical(
                    sample_kv_dir,
                    kv_tracer,
                    kv_sample_meta,
                    dtype_save=args.kv_dtype,
                    trajectory_idx=args.kv_trajectory_idx,
                )
            else:
                save_sample_kv(
                    sample_kv_dir,
                    kv_tracer,
                    kv_sample_meta,
                    dtype_save=args.kv_dtype,
                )
        else:
            pred = out
        ext = ".jpg"
        rel_saved = f"images/{row_idx:04d}_{safe_id}{ext}"

        if not args.no_save and not args.no_copy_images:
            dst = os.path.join(out_dir, rel_saved)
            if src_path:
                shutil.copy2(src_path, dst)
            else:
                image.save(dst, format="JPEG", quality=95)

        rec = {
            "item_id": f"eval_{row_idx}",
            "question_id": item.get("question_id"),
            "subset": args.subset,
            "relation": item.get("relation"),
            "question": qtext,
            "answer_options": options,
            "answer": item.get("answer"),
            "source_image_path": src_path,
            "image_from_base64": src_path is None,
            "prediction": pred,
            "option_in_context": args.option_in_context,
        }
        if not args.no_save:
            rec["saved_image"] = None if args.no_copy_images else rel_saved
        if sample_kv_dir is not None:
            rec["kv_trace_dir"] = sample_kv_dir
            rec["kv_layout"] = args.kv_layout
            rec["kv_trajectory_idx"] = args.kv_trajectory_idx

        results_rows.append(rec)

        if args.jsonl:
            # 不把超长 base64 打进 jsonl
            slim = {k: v for k, v in rec.items() if k != "question"}
            slim["question_preview"] = qtext[:200] + ("…" if len(qtext) > 200 else "")
            print(json.dumps(slim, ensure_ascii=False))
        else:
            print("-" * 60)
            print(f"item_id: eval_{row_idx}  question_id: {item.get('question_id')}")
            print(f"relation: {item.get('relation')}")
            print(f"Q: {qtext[:240]}{'…' if len(qtext) > 240 else ''}")
            print("-" * 60)
            print(pred)
            print()

    if not args.no_save:
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": out_dir,
            "emb_root": os.path.abspath(args.emb_root),
            "questions_json": qpath,
            "subset": args.subset,
            "image_dir": image_dir,
            "model_path": model_path,
            "option_in_context": args.option_in_context,
            "sft_question_idx": args.sft_question_idx,
            "no_cot": args.no_cot,
            "run_indices": run_indices,
            "no_copy_images": args.no_copy_images,
            "no_kv_trace": args.no_kv_trace,
            "kv_out_dir": kv_root,
            "kv_dtype": args.kv_dtype,
            "kv_layout": args.kv_layout,
            "kv_trajectory_idx": args.kv_trajectory_idx,
            "kv_model_meta": kv_model_meta,
        }
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results_rows, f, ensure_ascii=False, indent=2)
        if not args.jsonl:
            print("# 已写入 manifest.json、results.json", file=sys.stderr)
            if not args.no_copy_images:
                print(f"# 图片: {os.path.join(out_dir, 'images')}", file=sys.stderr)


if __name__ == "__main__":
    main()
