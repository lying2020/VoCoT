#!/usr/bin/env python3
"""
maze-dataset + VoCoT 本地推理试跑（测试样例）。

功能：
支持通过 --questions_json 读取现成样本（list[dict]，可含 viz_image_filename）；
使用 VoCoT/VolCano 做逐条推理，输出 results/manifest。
1. 先进入到 maze_vocot 
conda activate maze_vocot
2. 再调用脚本生成 maze 图和json文件
python generate_maze_eval_samples.py --out_dir /home1/cjl/MM_2026/maze_eval_tmp_volcano --n 32 --grid_n 9 --seed 0
3. 然后退出 maze_vocot 环境
conda deactivate
4. 之后进入 vocot 环境，开始推理测试
conda activate vocot
在 VoCoT 仓库根目录执行：
  python3 demo/run_maze_inference_demo.py
  python3 demo/run_maze_inference_demo.py --n 8 --start 0 --option_in_context
默认读取 MM_2026/maze_eval_tmp_volcano/maze_eval_samples.json 与 MM_2026/maze_eval_tmp_volcano/images/；
  现场生成样本请用: python3 demo/run_maze_inference_demo.py --generate_samples

每条结果在输出目录 images/ 下最多三份图（需未加 --no_save；原图/真值是否复制仍受 --no_copy_images 影响）：
  * 原图（无路径）：{index:04d}_{image_filename}
  * 真值路径图（若样本含 viz_image_filename）：{index:04d}_viz_*
  * 模型预测路径叠加图（随 prediction 解析成功写入，与 --no_copy_images 无关）：{index:04d}_prediction_path.png
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from PIL import Image

from model.load_model import infer, load_model
from project import volcano_7b_luoruipu1_path
from utils.maze_pred_viz import maze_image_with_predicted_path, parse_maze_path_from_text


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
    return os.path.join(_REPO_ROOT, "output", "maze", store, "inference_demo")


def _format_question_text(question: str, options: list[str] | None, option_in_context: bool) -> str:
    if not option_in_context or not options:
        return question
    return question + " Select from following options: " + "; ".join(options) + "."


def _default_maze_repo_dir() -> str:
    return os.path.abspath(os.path.join(_REPO_ROOT, "..", "maze-dataset"))


def _default_maze_eval_tmp_dir() -> str:
    """与 generate_maze_eval_samples.py --out_dir 常用路径一致：MM_2026/maze_eval_tmp_volcano """
    return os.path.abspath(os.path.join(_REPO_ROOT, "..", "maze_eval_tmp_volcano"))


def _default_maze_eval_questions_json() -> str:
    return os.path.join(_default_maze_eval_tmp_dir(), "maze_eval_samples.json")


def _default_maze_eval_image_dir() -> str:
    return os.path.join(_default_maze_eval_tmp_dir(), "images")


def _ensure_maze_importable(maze_repo_dir: str) -> None:
    if maze_repo_dir not in sys.path:
        sys.path.insert(0, maze_repo_dir)


def _build_single_sample(
    solved_maze: Any,
    image_filename: str,
    viz_image_filename: str | None,
) -> dict[str, Any]:
    start = tuple(int(x) for x in solved_maze.start_pos.tolist())
    end = tuple(int(x) for x in solved_maze.end_pos.tolist())
    sol = np.asarray(solved_maze.solution)
    path_cells = [[int(sol[i, 0]), int(sol[i, 1])] for i in range(sol.shape[0])]
    answer = json.dumps(path_cells, separators=(",", ":"))

    row: dict[str, Any] = {
        "question": (
            "The image is a maze: the green marker is the start, the red marker is the end, white cells are "
            "walkable, black is wall. Find a valid orthogonal path from start to end (only up/down/left/right "
            "between adjacent walkable cells, never through walls). Reply with the full path as a JSON array of "
            "[row, column] coordinates from the start cell to the end cell, e.g. [[3,3],[3,4],...]. Use only the "
            "maze image without a drawn solution path."
        ),
        "answer": answer,
        "image_filename": image_filename,
        "meta": {
            "task": "maze_path",
            "start_rc": list(start),
            "end_rc": list(end),
            "grid_n": int(solved_maze.grid_n),
            "path_len": int(sol.shape[0]),
            "path_cells": path_cells,
        },
    }
    if viz_image_filename:
        row["viz_image_filename"] = viz_image_filename
    return row


def _import_maze_dataset():
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "maze-dataset 在当前依赖链下需要 Python >= 3.10（例如 muutils 的 serializable_dataclass "
            "在 3.9 上不支持 kw_only）。\n"
            "建议：conda create -n maze_vocot python=3.10 -y && conda activate maze_vocot\n"
            "  按 VoCoT README 装好 torch 等后，执行: cd ../maze-dataset && pip install -e .\n"
            "  再回到 VoCoT 目录运行本脚本。\n"
            f"当前解释器: {sys.version}"
        )
    try:
        from maze_dataset import MazeDataset, MazeDatasetConfig
        from maze_dataset.generation import LatticeMazeGenerators
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "无法导入 maze_dataset。\n"
            "  • 官方 maze-dataset 需要 Python >= 3.10，且依赖含 jaxtyping、muutils、zanj 等。\n"
            "  • 推荐在 maze-dataset 仓库根目录执行: pip install -e .\n"
            "  • 最小补依赖可试: pip install 'jaxtyping>=0.2.19' 'muutils>=0.8.3' 'zanj>=0.5.0' matplotlib tqdm\n"
            f"  • 缺省模块: {getattr(e, 'name', '') or str(e)}"
        ) from e
    except ImportError as e:
        raise ImportError(
            "导入 maze_dataset 失败（常为依赖版本不匹配，例如 muutils 过旧缺少 _FORMAT_KEY）。\n"
            "  • 尝试: pip install -U 'muutils>=0.8.3' 'zanj>=0.5.0'\n"
            "  • 或在 maze-dataset 仓库根目录: pip install -e .\n"
            f"  • 原始错误: {e}"
        ) from e
    return MazeDataset, MazeDatasetConfig, LatticeMazeGenerators


def _generate_maze_samples(
    maze_repo_dir: str,
    save_dir: str,
    n: int,
    grid_n: int,
    seed: int | None,
    save_solution_viz: bool,
) -> tuple[str, str]:
    _ensure_maze_importable(maze_repo_dir)
    MazeDataset, MazeDatasetConfig, LatticeMazeGenerators = _import_maze_dataset()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    cfg = MazeDatasetConfig(
        name="vocot_maze_demo",
        grid_n=grid_n,
        n_mazes=max(1, n),
        maze_ctor=LatticeMazeGenerators.gen_dfs,
        maze_ctor_kwargs={},
    )
    dataset = MazeDataset.from_config(
        cfg,
        do_generate=True,
        load_local=False,
        save_local=False,
        do_download=False,
    )

    gen_dir = os.path.join(save_dir, "generated")
    img_dir = os.path.join(gen_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for i in range(min(n, len(dataset))):
        maze = dataset[i]
        pixels = maze.as_pixels(show_endpoints=True, show_solution=False)
        image_filename = f"maze_{i:04d}.png"
        image_path = os.path.join(img_dir, image_filename)
        Image.fromarray(pixels).save(image_path)
        viz_name: str | None = None
        if save_solution_viz:
            viz_name = f"maze_{i:04d}_with_path.png"
            pixels_viz = maze.as_pixels(show_endpoints=True, show_solution=True)
            Image.fromarray(pixels_viz).save(os.path.join(img_dir, viz_name))
        rows.append(_build_single_sample(maze, image_filename, viz_name))

    qpath = os.path.join(gen_dir, "maze_eval_samples.json")
    with open(qpath, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return qpath, img_dir


def main() -> None:
    p = argparse.ArgumentParser(description="maze-dataset + VoCoT 推理测试样例")
    _def_qjson = _default_maze_eval_questions_json()
    _def_imgdir = _default_maze_eval_image_dir()
    p.add_argument(
        "--questions_json",
        type=str,
        default=_def_qjson,
        help=f"样本 JSON（list；默认: {_def_qjson}）",
    )
    p.add_argument(
        "--image_dir",
        type=str,
        default=_def_imgdir,
        help=f"图像目录（默认: {_def_imgdir}）",
    )
    p.add_argument(
        "--generate_samples",
        action="store_true",
        help=(
            "在输出目录下用 maze-dataset 现场生成样本并推理（忽略 --questions_json/--image_dir 默认值；"
            "勿与 --no_save 同时使用）"
        ),
    )
    p.add_argument("--maze_repo_dir", type=str, default=_default_maze_repo_dir(), help="maze-dataset 仓库目录")
    p.add_argument("--n", type=int, default=8, help="跑多少条")
    p.add_argument("--start", type=int, default=0, help="起始下标（0-based）")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--grid_n", type=int, default=9, help="自动生成样本时迷宫边长")
    p.add_argument("--option_in_context", action="store_true")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--precision", type=str, default="fp16", choices=("fp16", "bf16", "fp32"))
    p.add_argument("--no_cot", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--jsonl", action="store_true")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--no_save", action="store_true")
    p.add_argument("--no_copy_images", action="store_true")
    p.add_argument(
        "--no_solution_viz",
        action="store_true",
        help="自动生成样本时不额外保存带解救路径的可视化图（默认保存 maze_XXXX_with_path.png）",
    )
    args = p.parse_args()

    model_path = _resolve_model_path(args.model_path)
    out_dir = os.path.abspath(args.output_dir) if args.output_dir else _default_output_dir(model_path)
    if not args.no_save:
        os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    if args.generate_samples:
        if args.no_save:
            raise ValueError("自动生成样本时请不要使用 --no_save（需要落盘临时样本）")
        qpath, image_dir = _generate_maze_samples(
            maze_repo_dir=os.path.abspath(args.maze_repo_dir),
            save_dir=out_dir,
            n=max(args.n + args.start, 1),
            grid_n=args.grid_n,
            seed=args.seed,
            save_solution_viz=not args.no_solution_viz,
        )
    else:
        qpath = os.path.abspath(args.questions_json)
        if not os.path.isfile(qpath):
            raise FileNotFoundError(
                f"找不到 questions JSON: {qpath}\n"
                "  可先按文档在 MM_2026/maze_eval_tmp_volcano 生成数据，或指定 --questions_json；"
                "也可加 --generate_samples 在输出目录现场生成。"
            )
        image_dir = os.path.abspath(args.image_dir)

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

    if not args.jsonl:
        print(f"# model_path: {model_path}", file=sys.stderr)
        print(f"# questions_json: {qpath}", file=sys.stderr)
        print(f"# image_dir: {image_dir}", file=sys.stderr)
        print(f"# 本批条数: {len(run_indices)}", file=sys.stderr)
        if not args.no_save:
            print(f"# 输出目录: {out_dir}", file=sys.stderr)

    model, preprocessor = load_model(model_path, device=args.device, precision=args.precision)

    results_rows: list[dict[str, Any]] = []
    for row_idx in run_indices:
        item = meta[row_idx]
        question = item["question"]
        options = item.get("answer_options")
        answer = item.get("answer")
        image_filename = item["image_filename"]
        query_text = _format_question_text(question, options, args.option_in_context)

        img_path = os.path.join(image_dir, os.path.basename(image_filename))
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"找不到图像: {img_path}")

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

        item_id = f"maze_{row_idx}"
        base_img = os.path.basename(image_filename)
        rel_saved = f"images/{row_idx:04d}_{base_img}"
        if not args.no_save and not args.no_copy_images:
            shutil.copy2(img_path, os.path.join(out_dir, rel_saved))

        item_meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
        start_rc = item_meta.get("start_rc")
        end_rc = item_meta.get("end_rc")
        grid_n_m = item_meta.get("grid_n")

        pred_path_cells = parse_maze_path_from_text(text)
        rel_pred_viz: str | None = None
        pred_viz_note: str | None = None
        if not args.no_save and pred_path_cells is not None:
            pred_img, pred_viz_note = maze_image_with_predicted_path(
                image,
                pred_path_cells,
                start_rc,
                end_rc,
                int(grid_n_m) if grid_n_m is not None else None,
            )
            if pred_img is not None:
                rel_pred_viz = f"images/{row_idx:04d}_prediction_path.png"
                pred_img.save(os.path.join(out_dir, rel_pred_viz))
        elif not args.no_save and pred_path_cells is None:
            pred_viz_note = "no_json_path_parsed"

        rec: dict[str, Any] = {
            "item_id": item_id,
            "index": row_idx,
            "question": question,
            "answer_options": options,
            "answer": answer,
            "image_filename": image_filename,
            "source_image_path": img_path,
            "prediction": text,
            "option_in_context": args.option_in_context,
            "prediction_path_cells": pred_path_cells,
            "saved_prediction_viz": rel_pred_viz,
            "prediction_viz_note": pred_viz_note,
        }
        if "meta" in item:
            rec["meta"] = item["meta"]
        if not args.no_save and not args.no_copy_images:
            rec["saved_image"] = rel_saved
        elif not args.no_save:
            rec["saved_image"] = None

        viz_fn = item.get("viz_image_filename")
        if viz_fn:
            viz_src = os.path.join(image_dir, os.path.basename(viz_fn))
            if os.path.isfile(viz_src):
                rec["viz_source_image_path"] = viz_src
                rel_viz = f"images/{row_idx:04d}_viz_{os.path.basename(viz_fn)}"
                if not args.no_save and not args.no_copy_images:
                    shutil.copy2(viz_src, os.path.join(out_dir, rel_viz))
                    rec["saved_viz_image"] = rel_viz
                elif not args.no_save:
                    rec["saved_viz_image"] = None

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
            "questions_json": os.path.abspath(qpath),
            "image_dir": os.path.abspath(image_dir),
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
            "maze_repo_dir": os.path.abspath(args.maze_repo_dir),
            "grid_n": args.grid_n,
            "generate_samples": args.generate_samples,
        }
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results_rows, f, ensure_ascii=False, indent=2)
        if not args.jsonl:
            print("# 已写入 manifest.json、results.json", file=sys.stderr)
            if not args.no_copy_images:
                print(f"# 图片副本: {os.path.join(out_dir, 'images')}", file=sys.stderr)
                if not args.no_solution_viz and args.generate_samples:
                    print(
                        "# 带路径可视化: 与推理用图同目录下 maze_*_with_path.png；已尽量复制到 output images/",
                        file=sys.stderr,
                    )


if __name__ == "__main__":
    main()
