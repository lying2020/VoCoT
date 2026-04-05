"""
从模型正文中解析迷宫路径 JSON，并在 maze-dataset 风格的 RGB 图上叠加 PATH（与 lattice 像素坐标一致）。
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from PIL import Image

# 与 maze_dataset.maze.lattice_maze.PixelColors 一致（避免强制依赖 maze_dataset 于 vocot 环境）
_WALL = (0, 0, 0)
_OPEN = (255, 255, 255)
_START = (0, 255, 0)
_END = (255, 0, 0)
_PATH = (0, 0, 255)


def parse_maze_path_from_text(text: str) -> list[list[int]] | None:
    """
    从正文抽取形如 [[r,c],[r,c],...] 的 JSON 数组（支持 CoT：取**最长**合法候选）。
    每项须为长度 2 的整数对；路径至少 2 个格点。
    """
    if not text or not text.strip():
        return None
    s = text.strip()
    try:
        data = json.loads(s)
        if _is_valid_path_list(data):
            return _normalize_path(data)
    except json.JSONDecodeError:
        pass

    candidates: list[list[list[int]]] = []
    n = len(s)
    i = 0
    while i < n:
        if s[i] != "[":
            i += 1
            continue
        depth = 0
        for j in range(i, n):
            if s[j] == "[":
                depth += 1
            elif s[j] == "]":
                depth -= 1
                if depth == 0:
                    chunk = s[i : j + 1]
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        i = j
                        break
                    if _is_valid_path_list(data):
                        candidates.append(_normalize_path(data))
                    i = j
                    break
        i += 1

    if not candidates:
        return None
    return max(candidates, key=len)


def _is_valid_path_list(data: Any) -> bool:
    if not isinstance(data, list) or len(data) < 2:
        return False
    for cell in data:
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            return False
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in cell):
            return False
    return True


def _normalize_path(data: list[Any]) -> list[list[int]]:
    return [[int(c[0]), int(c[1])] for c in data]


def expected_maze_pixel_side(grid_n: int) -> int:
    return int(grid_n) * 2 + 1


def overlay_path_on_maze_rgb(
    rgb: np.ndarray,
    path_cells: list[list[int]],
    start_rc: list[int] | tuple[int, int] | None,
    end_rc: list[int] | tuple[int, int] | None,
    grid_n: int | None,
) -> tuple[np.ndarray | None, str | None]:
    """
    在 maze 渲染图（无路径、含起终点）上叠蓝色 PATH。坐标规则与 maze-dataset SolvedMaze.as_pixels 一致。

    返回 (out_rgb 或 None, 错误信息或 None)。
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None, "image must be HxWx3 RGB"
    if not path_cells or len(path_cells) < 2:
        return None, "empty path"

    h, w = rgb.shape[0], rgb.shape[1]
    if grid_n is not None:
        exp = expected_maze_pixel_side(int(grid_n))
        if h != exp or w != exp:
            return None, f"shape ({h},{w}) != ({exp},{exp}) for grid_n={grid_n}"

    out = rgb.copy()

    for coord in path_cells:
        r, c = int(coord[0]), int(coord[1])
        if r < 0 or c < 0 or r * 2 + 1 >= h or c * 2 + 1 >= w:
            return None, f"path cell out of bounds: ({r},{c})"
        out[r * 2 + 1, c * 2 + 1] = _PATH

    for idx in range(len(path_cells) - 1):
        coord = path_cells[idx]
        next_c = path_cells[idx + 1]
        dr = int(next_c[0]) - int(coord[0])
        dc = int(next_c[1]) - int(coord[1])
        if abs(dr) + abs(dc) != 1:
            return None, f"non-adjacent step {(coord)} -> {next_c}"
        pr = int(coord[0]) * 2 + 1 + dr
        pc = int(coord[1]) * 2 + 1 + dc
        if pr < 0 or pc < 0 or pr >= h or pc >= w:
            return None, f"connector pixel out of bounds ({pr},{pc})"
        out[pr, pc] = _PATH

    if start_rc is not None and len(start_rc) >= 2:
        sr, sc = int(start_rc[0]), int(start_rc[1])
        out[sr * 2 + 1, sc * 2 + 1] = _START
    if end_rc is not None and len(end_rc) >= 2:
        er, ec = int(end_rc[0]), int(end_rc[1])
        out[er * 2 + 1, ec * 2 + 1] = _END

    return out, None


def overlay_path_on_maze_rgb_loose(
    rgb: np.ndarray,
    path_cells: list[list[int]],
    start_rc: list[int] | tuple[int, int] | None,
    end_rc: list[int] | tuple[int, int] | None,
    grid_n: int | None,
) -> tuple[np.ndarray | None, str | None]:
    """仅绘制格心 PATH；相邻步之间若正交则补走廊像素，否则跳过该段（用于模型路径不完全合法时的对照图）。"""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None, "image must be HxWx3 RGB"
    if not path_cells:
        return None, "empty path"
    h, w = rgb.shape[0], rgb.shape[1]
    if grid_n is not None:
        exp = expected_maze_pixel_side(int(grid_n))
        if h != exp or w != exp:
            return None, f"shape ({h},{w}) != ({exp},{exp}) for grid_n={grid_n}"

    out = rgb.copy()
    for coord in path_cells:
        r, c = int(coord[0]), int(coord[1])
        if r < 0 or c < 0 or r * 2 + 1 >= h or c * 2 + 1 >= w:
            continue
        out[r * 2 + 1, c * 2 + 1] = _PATH

    for idx in range(len(path_cells) - 1):
        coord = path_cells[idx]
        next_c = path_cells[idx + 1]
        dr = int(next_c[0]) - int(coord[0])
        dc = int(next_c[1]) - int(coord[1])
        if abs(dr) + abs(dc) != 1:
            continue
        pr = int(coord[0]) * 2 + 1 + dr
        pc = int(coord[1]) * 2 + 1 + dc
        if 0 <= pr < h and 0 <= pc < w:
            out[pr, pc] = _PATH

    if start_rc is not None and len(start_rc) >= 2:
        sr, sc = int(start_rc[0]), int(start_rc[1])
        if 0 <= sr * 2 + 1 < h and 0 <= sc * 2 + 1 < w:
            out[sr * 2 + 1, sc * 2 + 1] = _START
    if end_rc is not None and len(end_rc) >= 2:
        er, ec = int(end_rc[0]), int(end_rc[1])
        if 0 <= er * 2 + 1 < h and 0 <= ec * 2 + 1 < w:
            out[er * 2 + 1, ec * 2 + 1] = _END

    return out, None


def maze_image_with_predicted_path(
    image: Image.Image,
    path_cells: list[list[int]],
    start_rc: list[int] | tuple[int, int] | None,
    end_rc: list[int] | tuple[int, int] | None,
    grid_n: int | None,
) -> tuple[Image.Image | None, str | None]:
    rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    out, err = overlay_path_on_maze_rgb(rgb, path_cells, start_rc, end_rc, grid_n)
    if out is None:
        out2, err2 = overlay_path_on_maze_rgb_loose(rgb, path_cells, start_rc, end_rc, grid_n)
        if out2 is None:
            return None, err or err2 or "overlay failed"
        return Image.fromarray(out2), f"loose_overlay({err})"
    return Image.fromarray(out), None
