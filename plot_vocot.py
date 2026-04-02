import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
from typing import Any, List, Optional, Union
import colorsys


def _hex_to_rgb01(hex_color: str):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return r, g, b


def _blend_with_black(color_rgb, t: float):
    """t in [0,1]: 0 means original color, 1 means black."""
    r, g, b = color_rgb
    return (r * (1.0 - t), g * (1.0 - t), b * (1.0 - t))


def _reduce_saturation(color_rgb, saturation_scale: float):
    """通过缩放 HSV 空间的饱和度，让颜色“更淡/不那么艳”。"""
    r, g, b = color_rgb
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = max(0.0, min(1.0, s * saturation_scale))
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (r2, g2, b2)


def _pick_font(preferred: List[str]) -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return "DejaVu Serif"


def plot_vocot_grid(
    num_row: int,
    num_col: int,
    # 左边紫色部分：点数/均值/方差，均为 (num_row, num_col) 的二维 list/array
    left_counts: Union[List[List[int]], np.ndarray],
    left_means: Union[List[List[float]], np.ndarray],
    left_stds: Union[List[List[float]], np.ndarray],
    # 右边绿色部分：点数/均值/方差/最大y值（用于截断），均为 (num_row, num_col) 的二维 list/array
    right_counts: Union[List[List[int]], np.ndarray],
    right_means: Union[List[List[float]], np.ndarray],
    right_stds: Union[List[List[float]], np.ndarray],
    right_max_y: Union[List[List[float]], np.ndarray],
    seed: int = 0,
    save_path: Optional[str] = None,
    bar_width: float = 0.9,
    y_clip_min: float = 0.0,
    color_saturation_scale: float = 0.65,
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    label_font: Optional[str] = None,
):
    """
    复现目标示例图的“网格 + 每行配色渐深 + 每子图高斯噪声柱状条”。
    不包含 legend / 坐标轴标签。

    约定：每个子图都是：
      左段(紫)：从 `left_*` 生成 left_counts 个高斯噪声点
      右段(绿)：从 `right_*` 生成 right_counts 个高斯噪声点
    并将整体 y 值裁剪到 `[y_clip_min, right_max_y[r][c]]`。

    画法：每个点用竖直线段表示，线段底部固定在 y=0，
    顶部到对应的 y 值（而不是柱状图填充）。
    """
    def _as_2d(name: str, v: Any) -> np.ndarray:
        arr = np.asarray(v)
        if arr.shape != (num_row, num_col):
            raise ValueError(f"{name} 需要形状为 ({num_row}, {num_col})，实际为 {arr.shape}")
        return arr

    left_counts = _as_2d("left_counts", left_counts).astype(int)
    left_means = _as_2d("left_means", left_means).astype(float)
    left_stds = _as_2d("left_stds", left_stds).astype(float)
    right_counts = _as_2d("right_counts", right_counts).astype(int)
    right_means = _as_2d("right_means", right_means).astype(float)
    right_stds = _as_2d("right_stds", right_stds).astype(float)
    right_max_y = _as_2d("right_max_y", right_max_y).astype(float)

    # 行方向：颜色逐渐加深（top -> bottom）
    # 紫色（左段）：浅紫 -> 深紫红
    purple_light = ["#d6c2ff", "#c9b0ff", "#b69bff", "#9f86ff", "#8b72ff"]
    purple_deep = ["#7a2cff", "#6b20e0", "#5c17c4", "#4c10a8", "#3d0b8f"]
    # 绿色（右段）：浅绿 -> 深绿色
    green_light = ["#baf7c9", "#a2f0b4", "#8be889", "#72dc6e", "#59c454"]
    green_deep = ["#1f9a4d", "#17843f", "#116f35", "#0b5a2a", "#06491f"]
    if num_row > len(purple_light):
        raise ValueError(f"num_row 最大支持 {len(purple_light)}，当前={num_row}")
    purple_light = purple_light[:num_row]
    purple_deep = purple_deep[:num_row]
    green_light = green_light[:num_row]
    green_deep = green_deep[:num_row]

    rng = np.random.default_rng(seed)

    # 画布大小按示例图风格：紧凑但不挤压
    fig, axes = plt.subplots(num_row, num_col, figsize=(14, 7), squeeze=False)
    fig.patch.set_facecolor("white")

    # 先生成并缓存所有子图数据，计算全局 y_max/xlim（不裁剪分数）
    # y_max = 所有数据中最大的前100个点的平均值 + 0.05
    # xlim = 使用所有子图里最大的点数 max_total_n 统一对齐
    cached = [[None for _ in range(num_col)] for _ in range(num_row)]
    all_y_list: List[np.ndarray] = []
    max_total_n = 0

    for r in range(num_row):
        # 每行都使用一组“首尾端点”颜色；线性渐变会从浅 -> 深
        p_light = _hex_to_rgb01(purple_light[r])
        p_deep = _hex_to_rgb01(purple_deep[r])
        g_light = _hex_to_rgb01(green_light[r])
        g_deep = _hex_to_rgb01(green_deep[r])

        # 为了更贴近“不断加深”的直觉，再额外做一次轻微压暗（不会改变端点层次）
        t_dark = 0.35 * (r / max(1, num_row - 1))
        p_light = _blend_with_black(p_light, t_dark)
        p_deep = _blend_with_black(p_deep, t_dark)
        g_light = _blend_with_black(g_light, t_dark)
        g_deep = _blend_with_black(g_deep, t_dark)

        # 降饱和度：让颜色“更淡”
        p_light = _reduce_saturation(p_light, color_saturation_scale)
        p_deep = _reduce_saturation(p_deep, color_saturation_scale)
        g_light = _reduce_saturation(g_light, color_saturation_scale)
        g_deep = _reduce_saturation(g_deep, color_saturation_scale)

        purple_cmap = LinearSegmentedColormap.from_list(f"purple_r{r}", [p_light, p_deep])
        green_cmap = LinearSegmentedColormap.from_list(f"green_r{r}", [g_light, g_deep])

        for c in range(num_col):
            ax = axes[r, c]

            left_n = int(left_counts[r, c])
            right_n = int(right_counts[r, c])
            total_n = left_n + right_n
            if total_n <= 0:
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # 左段 / 右段高斯噪声
            y_left = rng.normal(
                loc=float(left_means[r, c]),
                scale=max(0.0, float(left_stds[r, c])),
                size=left_n,
            )
            y_right = rng.normal(
                loc=float(right_means[r, c]),
                scale=max(0.0, float(right_stds[r, c])),
                size=right_n,
            )
            y = np.concatenate([y_left, y_right], axis=0)

            # 不裁剪重要性分数；但 y 轴下界固定为 0，绘图时从 y=0 作为基线画竖线。
            all_y_list.append(y)

            # 颜色：左段紫渐变，右段绿渐变
            bar_colors = np.zeros((total_n, 4), dtype=float)
            if left_n > 0:
                bar_colors[:left_n] = purple_cmap(np.linspace(0, 1, left_n))
            if right_n > 0:
                bar_colors[left_n:] = green_cmap(np.linspace(0, 1, right_n))
            if left_n - 1 >= 0 and right_n > 0:
                bar_colors[left_n - 1] = 0.5 * (
                    np.array(purple_cmap(1.0)) + np.array(green_cmap(0.0))
                )

            cached[r][c] = (y, bar_colors, total_n)
            if total_n > max_total_n:
                max_total_n = total_n

            # 让各子图背景一致、边框淡一些（更像拼接的图组）
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color("#d0d0d0")
            ax.set_facecolor("white")

    # 统一 ylim=[0, y_max]，其中 y_max = 最大前100个点的均值 + 0.05
    if len(all_y_list) == 0:
        y_max = 0.05
    else:
        all_y = np.concatenate(all_y_list, axis=0)
        k = int(max(1, min(100, all_y.size)))
        top_vals = np.partition(all_y, -k)[-k:]
        y_max = float(np.mean(top_vals) + 0.05)
    if not np.isfinite(y_max) or y_max <= 0.0:
        y_max = 0.05

    if max_total_n <= 0:
        max_total_n = 1

    # 第二遍：绘制 vlines 并统一设定坐标范围
    for r in range(num_row):
        for c in range(num_col):
            ax = axes[r, c]
            item = cached[r][c]
            if item is None:
                continue
            y, bar_colors, total_n = item

            x = np.arange(total_n)
            colors = [tuple(col) for col in bar_colors]
            ax.vlines(x, 0.0, y, colors=colors, linewidth=bar_width)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-0.5, max_total_n - 0.5)
            ax.set_ylim(0.0, y_max)

    # 行/列标签（图例/轴标签仍不显示）
    if row_labels is None:
        row_labels = [rf"$h_{{{i}}}$" for i in range(num_row)]
    if col_labels is None:
        col_labels = [rf"$l_{{{i}}}$" for i in range(num_col)]
    row_labels = row_labels[:num_row]
    col_labels = col_labels[:num_col]
    if label_font is None:
        label_font = _pick_font(
            ["Times New Roman", "Times", "Nimbus Roman", "Nimbus Roman No9 L", "Liberation Serif"]
        )

    # 基于 axes 在 fig 坐标系的位置放置文本
    left_pad = 0.015
    top_pad = 0.01
    for r in range(num_row):
        bb = axes[r, 0].get_position()
        fig.text(
            bb.x0 - left_pad,
            (bb.y0 + bb.y1) * 0.5,
            row_labels[r],
            ha="right",
            va="center",
            fontfamily=label_font,
            fontsize=20,
            fontweight="bold",
        )
    for c in range(num_col):
        bb = axes[0, c].get_position()
        fig.text(
            (bb.x0 + bb.x1) * 0.5,
            bb.y1 + top_pad,
            col_labels[c],
            ha="center",
            va="bottom",
            fontfamily=label_font,
            fontsize=20,
            fontweight="bold",
        )

    plt.subplots_adjust(wspace=0.08, hspace=0.28)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def main():
    # 最大支持 5x7；也可以改成 4x6 等更小的网格
    num_row, num_col = 4, 6
    seed = 0

    # 下面这些默认参数已经“展开成具体的 5*7 数值”，方便你直接逐格手动微调。
    # 这组数值对应当前 seed=0 时自动生成出来的结果（已固化）。
    left_counts = [
        [101, 101, 101, 101, 101, 101, 101],
        [107, 107, 107, 107, 107, 107, 107],
        [113, 113, 113, 113, 113, 113, 113],
        [108, 108, 108, 108, 108, 108, 108],
        [123, 123, 123, 123, 123, 123, 123],
    ]
    right_counts = [
        [157, 157, 157, 157, 157, 157, 157],
        [167, 167, 167, 167, 167, 167, 167],
        [159, 159, 159, 159, 159, 159, 159],
        [176, 176, 176, 176, 176, 176, 176],
        [160, 160, 160, 160, 160, 160, 160],
    ]

    left_means = [
        [0.500000, 0.515000, 0.531000, 0.548000, 0.566000, 0.582000, 0.601000],
        [0.534482, 0.549482, 0.565482, 0.582482, 0.600482, 0.616482, 0.635482],
        [0.575948, 0.590948, 0.606948, 0.623948, 0.641948, 0.657948, 0.676948],
        [0.606849, 0.612349, 0.637849, 0.654849, 0.672849, 0.688849, 0.707849],
        [0.660000, 0.675000, 0.691000, 0.708000, 0.726000, 0.742000, 0.761000],
    ]
    right_means = [
        [0.619486, 0.600466, 0.567339, 0.171406, 0.153614, 0.140376, 0.107026],
        [0.587021, 0.568001, 0.534874, 0.168941, 0.141579, 0.127911, 0.104561],
        [0.566656, 0.547636, 0.514509, 0.148576, 0.120784, 0.107546, 0.094196],
        [0.555384, 0.536364, 0.503237, 0.137304, 0.109512, 0.096274, 0.096924],
        [0.539486, 0.520466, 0.487339, 0.121406, 0.113614, 0.110376, 0.107026],
    ]

    left_stds = [
        [0.320000, 0.320000, 0.320000, 0.320000, 0.320000, 0.320000, 0.320000],
        [0.360000, 0.360000, 0.360000, 0.360000, 0.360000, 0.360000, 0.360000],
        [0.360000, 0.360000, 0.360000, 0.360000, 0.360000, 0.360000, 0.360000],
        [0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000, 0.400000],
        [0.416000, 0.416000, 0.416000, 0.416000, 0.416000, 0.416000, 0.416000],
    ]
    right_stds = [
        [0.412991, 0.400311, 0.378226, 0.1200937, 0.122409, 0.110251, 0.118017],
        [0.465450, 0.450369, 0.404103, 0.116044, 0.121108, 0.114853, 0.128410],
        [0.466025, 0.450383, 0.403139, 0.104432, 0.111576, 0.124240, 0.116813],
        [0.418181, 0.400435, 0.369527, 0.121408, 0.125477, 0.124466, 0.113350],
        [0.439486, 0.420466, 0.347339, 0.121406, 0.113614, 0.110376, 0.110000],
    ]

    right_max_y = [
        [2.064953, 2.001553, 1.891131, 1.004687, 0.912046, 0.801253, 0.690087],
        [2.216097, 2.144294, 2.019235, 1.015295, 0.910375, 0.784897, 0.658996],
        [2.197745, 2.123977, 1.995497, 0.964089, 0.856299, 0.727388, 0.598041],
        [2.369016, 2.287885, 2.146582, 1.012231, 0.893682, 0.751904, 0.609648],
        [2.427686, 2.342096, 2.193026, 0.996327, 0.871262, 0.721692, 0.582026],
    ]

    # 如果 num_row/num_col 小于 5/7，这里会自动切片到对应大小，避免报错
    left_counts = [row[:num_col] for row in left_counts[:num_row]]
    right_counts = [row[:num_col] for row in right_counts[:num_row]]
    left_means = [row[:num_col] for row in left_means[:num_row]]
    right_means = [row[:num_col] for row in right_means[:num_row]]
    left_stds = [row[:num_col] for row in left_stds[:num_row]]
    right_stds = [row[:num_col] for row in right_stds[:num_row]]
    right_max_y = [row[:num_col] for row in right_max_y[:num_row]]

    out_path = os.path.join(os.path.dirname(__file__), "plot_vocot.png")
    plot_vocot_grid(
        num_row=num_row,
        num_col=num_col,
        left_counts=left_counts,
        left_means=left_means,
        left_stds=left_stds,
        right_counts=right_counts,
        right_means=right_means,
        right_stds=right_stds,
        right_max_y=right_max_y,
        seed=seed,
        save_path=out_path,
        bar_width=0.9,
        y_clip_min=0.0,
        color_saturation_scale=0.65,
        row_labels=[rf"$h_{{{i}}}$" for i in range(num_row)],
        col_labels=[r"$l_{0}$", r"$l_{5}$", r"$l_{10}$", r"$l_{20}$", r"$l_{25}$", r"$l_{30}$", r"$l_{35}$"][:num_col],
        label_font=None,
    )


if __name__ == "__main__":
    main()

