#!/usr/bin/env python3
"""
将 EmbSpatial 的 JSON（bench / sft）中的 base64 图像解码为独立文件，并导出元数据（不含内联图像）。

默认读取 project.embspatial_path 下的 embspatial_bench.json、embspatial_sft.json。
若条目的 image 已是磁盘路径，则复制到输出目录（同一相对命名规则）。

用法（在 VoCoT 仓库根目录）:
  python demo/export_embspatial_dataset.py --subset all
  python demo/export_embspatial_dataset.py --subset bench --out /data/embspatial_export
  python demo/export_embspatial_dataset.py --subset sft --image_dir /path/to/extra  # 仅文件名时在此查找原图
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import shutil
import sys
from io import BytesIO

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from PIL import Image

from project import embspatial_path


def _safe_slug(s: str, max_len: int = 120) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(s))[:max_len]


def _load_json_list(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} 须为非空 JSON 数组")
    return data


def _bench_filter(meta: list[dict]) -> list[dict]:
    """与 EmbSpatial demo / EmbSpatialDataset 一致，去掉两种 relation。"""
    return [x for x in meta if x.get("relation") not in ("on top of", "inside")]


def _write_image_bytes(dst: str, raw_bytes: bytes) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    im = Image.open(BytesIO(raw_bytes)).convert("RGB")
    im.save(dst, format="JPEG", quality=95)


def _export_one_image(
    raw_image: str,
    image_dir: str | None,
    dst_jpg: str,
) -> tuple[str, str]:
    """
    将单条 image 字段落到 dst_jpg。
    返回 (方式, 详情): ("file"|"image_dir"|"base64"|"decode_error", ...)
    """
    raw = (raw_image or "").strip()
    if not raw:
        raise ValueError("空 image 字段")

    if os.path.isfile(raw):
        shutil.copy2(raw, dst_jpg)
        return "file", raw

    if image_dir:
        p = os.path.join(image_dir, os.path.basename(raw))
        if os.path.isfile(p):
            shutil.copy2(p, dst_jpg)
            return "image_dir", p

    try:
        data = base64.b64decode(raw, validate=False)
    except binascii.Error as e:
        return "decode_error", str(e)

    try:
        _write_image_bytes(dst_jpg, data)
    except (OSError, ValueError) as e:
        return "decode_error", f"非有效图像字节: {e}"
    return "base64", f"decoded {len(data)} bytes"


def export_subset(
    *,
    subset: str,
    json_path: str,
    out_root: str,
    image_dir: str | None,
    bench_filter: bool,
) -> dict:
    meta = _load_json_list(json_path)
    if bench_filter:
        meta = _bench_filter(meta)

    sub_out = os.path.join(out_root, subset)
    img_dir = os.path.join(sub_out, "images")
    os.makedirs(img_dir, exist_ok=True)

    records: list[dict] = []
    stats = {"ok": 0, "fail": 0, "by_source": {}}

    for idx, item in enumerate(meta):
        qid = item.get("question_id", f"row_{idx}")
        slug = _safe_slug(qid)
        rel_img = f"images/{idx:05d}_{slug}.jpg"
        abs_img = os.path.join(sub_out, rel_img)

        rec = json.loads(json.dumps(item, ensure_ascii=False))
        raw_img = item.get("image", "")
        how, detail = _export_one_image(raw_img, image_dir, abs_img)
        stats["by_source"][how] = stats["by_source"].get(how, 0) + 1
        if how == "decode_error":
            stats["fail"] += 1
            rec["image_export_error"] = detail
            rec["image_relative"] = None
        else:
            stats["ok"] += 1
            rec["image_relative"] = rel_img.replace("\\", "/")
            rec["image_export_source"] = how
            if how in ("file", "image_dir"):
                rec["image_source_path"] = detail

        # 元数据里不再保留巨大 base64
        if "image" in rec and isinstance(rec["image"], str) and len(rec["image"]) > 500:
            rec["image"] = None
            rec["image_was_inline_base64"] = True
        elif "image" in rec and isinstance(rec["image"], str):
            rec["image_was_inline_base64"] = False

        records.append(rec)

    meta_path = os.path.join(sub_out, "records.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    summary = {
        "subset": subset,
        "source_json": os.path.abspath(json_path),
        "count": len(records),
        "images_dir": os.path.join(sub_out, "images"),
        "records_json": meta_path,
        **stats,
    }
    with open(os.path.join(sub_out, "export_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="导出 EmbSpatial 图像与元数据")
    p.add_argument(
        "--subset",
        choices=("bench", "sft", "all"),
        default="all",
        help="导出 bench / sft / 两者",
    )
    p.add_argument(
        "--emb_root",
        type=str,
        default=embspatial_path,
        help="embspatial 目录（内含 embspatial_*.json）",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出根目录；默认 <emb_root>/exported",
    )
    p.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="若 JSON 里 image 仅为文件名，可在此目录查找原图",
    )
    p.add_argument(
        "--no_bench_filter",
        action="store_true",
        help="bench 也导出 relation 为 on top of / inside 的样本（与默认评测集不一致）",
    )
    args = p.parse_args()

    emb_root = os.path.abspath(args.emb_root)
    out_root = os.path.abspath(args.out) if args.out else os.path.join(emb_root, "exported")
    image_dir = os.path.abspath(args.image_dir) if args.image_dir else None

    todo = []
    if args.subset in ("bench", "all"):
        todo.append(
            (
                "bench",
                os.path.join(emb_root, "embspatial_bench.json"),
                not args.no_bench_filter,
            )
        )
    if args.subset in ("sft", "all"):
        todo.append(("sft", os.path.join(emb_root, "embspatial_sft.json"), False))

    manifest = {
        "emb_root": emb_root,
        "out_root": out_root,
        "image_dir": image_dir,
        "subsets": [],
    }

    for subset, jpath, bench_filter in todo:
        if not os.path.isfile(jpath):
            raise FileNotFoundError(f"找不到: {jpath}")
        print(f"# 导出 {subset} <- {jpath}", file=sys.stderr)
        summ = export_subset(
            subset=subset,
            json_path=jpath,
            out_root=out_root,
            image_dir=image_dir,
            bench_filter=bench_filter,
        )
        manifest["subsets"].append(summ)
        print(
            f"#   条数 {summ['count']} 成功写图 {summ['ok']} 失败 {summ['fail']}",
            file=sys.stderr,
        )
        print(f"#   记录: {summ['records_json']}", file=sys.stderr)

    man_path = os.path.join(out_root, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"# 总清单: {man_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
