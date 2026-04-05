#!/usr/bin/env python3
"""
用 maze-dataset 生成 VoCoT 评测用的迷宫图片 + questions_json（不依赖 VoCoT/torch）。

建议在 Python>=3.10 的环境里运行（例如 conda env: maze_vocot）：
  python generate_maze_eval_samples.py --out_dir /tmp/maze_eval --n 32 --grid_n 9 --seed 0
  python generate_maze_eval_samples.py --out_dir /tmp/maze_eval --n 2000 --grid_ns 9 13 17 --seed 0
  python generate_maze_eval_samples.py --out_dir /tmp/maze_eval --n 2000 --grid_ns_csv 9,13,17 --seed 0

输出：
  <out_dir>/
    images/
      maze_0000.png                 # 给模型的输入图（无路径）
      maze_0000_with_path.png        # 人工对照图（带路径，可选）
    maze_eval_samples.json           # list[dict]：路径规划题；answer 为 path 的 JSON 数组字符串；meta.path_cells 为 GT
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Any

import numpy as np
from PIL import Image


def _ensure_maze_importable(maze_repo_dir: str | None) -> None:
    if not maze_repo_dir:
        return
    maze_repo_dir = os.path.abspath(maze_repo_dir)
    if maze_repo_dir not in sys.path:
        sys.path.insert(0, maze_repo_dir)


def _build_single_sample(solved_maze: Any, image_filename: str, viz_image_filename: str | None) -> dict[str, Any]:
    start = tuple(int(x) for x in solved_maze.start_pos.tolist())
    end = tuple(int(x) for x in solved_maze.end_pos.tolist())
    sol = np.asarray(solved_maze.solution)
    path_cells = [[int(sol[i, 0]), int(sol[i, 1])] for i in range(sol.shape[0])]
    # 与评分脚本对齐：紧凑 JSON 字符串，无额外空格
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


def _parse_grid_ns(grid_ns_list: list[int] | None, grid_ns_csv: str | None, grid_n_single: int) -> list[int]:
    if grid_ns_list:
        grid_ns = [int(x) for x in grid_ns_list]
    elif grid_ns_csv:
        grid_ns = [int(x.strip()) for x in grid_ns_csv.split(",") if x.strip()]
    else:
        grid_ns = [int(grid_n_single)]
    grid_ns = [x for x in grid_ns if x > 1]
    if not grid_ns:
        raise ValueError("grid_n 需要 >=2（例如 9 / 13 / 17）")
    # 去重但保持顺序
    seen: set[int] = set()
    uniq: list[int] = []
    for x in grid_ns:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def main() -> None:
    p = argparse.ArgumentParser(description="生成 maze-dataset -> VoCoT 评测样本（图片+JSON）")
    p.add_argument("--maze_repo_dir", type=str, default=None, help="maze-dataset 仓库目录（可选；不用 pip 安装时用）")
    p.add_argument("--out_dir", type=str, default="/home1/cjl/MM_2026/maze_eval_tmp", help="输出目录（默认 ./output/maze_eval_samples/<timestamp>）")
    p.add_argument("--n", type=int, default=3200, help="生成多少条样本")
    p.add_argument("--grid_n", type=int, default=9, help="迷宫边长（单值；也可用 --grid_ns / --grid_ns_csv）")
    p.add_argument(
        "--grid_ns",
        type=int,
        nargs="+",
        default=None,
        help="迷宫边长列表（空格分隔），例如：--grid_ns 9 13 17；优先于 --grid_ns_csv/--grid_n",
    )
    p.add_argument(
        "--grid_ns_csv",
        type=str,
        default=None,
        help="迷宫边长列表（逗号分隔），例如：--grid_ns_csv 9,13,17；优先于 --grid_n",
    )
    p.add_argument(
        "--per_grid",
        type=int,
        default=None,
        help="每个 grid_n 生成多少条；若设置，则总条数约为 per_grid * len(grid_ns)，并覆盖 --n",
    )
    p.add_argument("--seed", type=int, default=0, help="随机种子")
    p.add_argument("--no_solution_viz", action="store_true", help="不保存带路径可视化图")
    args = p.parse_args()

    _ensure_maze_importable(args.maze_repo_dir)

    if sys.version_info < (3, 10):
        raise RuntimeError(f"需要 Python>=3.10，当前: {sys.version}")

    try:
        from maze_dataset import MazeDataset, MazeDatasetConfig
        from maze_dataset.generation import LatticeMazeGenerators
    except Exception as e:
        raise RuntimeError(
            "无法导入 maze_dataset。请在 Python>=3.10 环境里执行：\n"
            "  - 若你在 maze-dataset 仓库：pip install -e .\n"
            "  - 或 pip install maze-dataset\n"
            f"原始错误: {e}"
        ) from e

    random.seed(args.seed)
    np.random.seed(args.seed)

    grid_ns = _parse_grid_ns(
        grid_ns_list=args.grid_ns,
        grid_ns_csv=args.grid_ns_csv,
        grid_n_single=args.grid_n,
    )
    if args.per_grid is not None:
        if args.per_grid <= 0:
            raise ValueError("--per_grid 需要 > 0")
        total_n = int(args.per_grid) * len(grid_ns)
    else:
        total_n = int(args.n)
    if total_n <= 0:
        raise ValueError("--n 需要 > 0")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(args.out_dir or os.path.join(os.getcwd(), "output", "maze_eval_samples", ts))
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows: list[dict[str, Any]] = []
    idx_out = 0
    for g in grid_ns:
        if args.per_grid is not None:
            n_g = int(args.per_grid)
        else:
            # 均匀分配，余数给前面的 grid
            base = total_n // len(grid_ns)
            rem = total_n % len(grid_ns)
            n_g = base + (1 if grid_ns.index(g) < rem else 0)
        if n_g <= 0:
            continue

        cfg = MazeDatasetConfig(
            name=f"vocot_maze_eval_g{g}",
            grid_n=int(g),
            n_mazes=int(max(n_g, 1)),
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

        for i in range(min(n_g, len(dataset))):
            maze = dataset[i]
            img_name = f"maze_{idx_out:06d}_g{g}.png"
            Image.fromarray(maze.as_pixels(show_endpoints=True, show_solution=False)).save(os.path.join(img_dir, img_name))

            viz_name: str | None = None
            if not args.no_solution_viz:
                viz_name = f"maze_{idx_out:06d}_g{g}_with_path.png"
                Image.fromarray(maze.as_pixels(show_endpoints=True, show_solution=True)).save(os.path.join(img_dir, viz_name))

            rows.append(_build_single_sample(maze, img_name, viz_name))
            idx_out += 1

    qpath = os.path.join(out_dir, "maze_eval_samples.json")
    with open(qpath, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(out_dir)
    print(qpath)
    print(img_dir)


if __name__ == "__main__":
    main()

