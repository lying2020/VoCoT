# maze-dataset 与 VoCoT 迷宫推理快速开始（中文）

本文说明 [**maze-dataset**](https://github.com/understanding-search/maze-dataset) 仓库是做什么的、目录里各块负责什么、如何安装依赖、以及如何在 **VoCoT** 里用 `run_maze_inference_demo.py` 做 **demo 推理**并查看 **路径可视化**。

---

## 一、maze-dataset 是做什么的？

**maze-dataset** 是一个用于 **程序化生成、求解、过滤、可视化** 迷宫数据集的 Python 库，主要面向 **训练或评测机器学习模型**（例如迷宫 Transformer 的可解释性研究 [maze-transformer](https://github.com/understanding-search/maze-transformer)）。

典型能力包括：

| 能力 | 说明 |
|------|------|
| **生成** | 多种算法：DFS、Wilson、渗流（percolation）等；可配置网格大小与数量 |
| **求解** | 得到 `SolvedMaze`：含从起点到终点的路径序列 |
| **过滤** | 按路径长度、复杂度、自定义属性筛选子集 |
| **可视化** | 栅格像素图 `as_pixels`、或使用 `plotting` 中的绘图工具 |
| **向量化 / 导出** | 多种 tokenization 与序列格式，供 NLP/序列模型训练 |

官方文档与论文：[在线文档](https://understanding-search.github.io/maze-dataset/)，[arXiv:2309.10498](http://arxiv.org/abs/2309.10498)。

---

## 二、仓库目录结构（重点关注 `maze_dataset/`）

以仓库根目录为参考：

| 路径 | 作用 |
|------|------|
| **`maze_dataset/maze/`** | 核心迷宫对象：`LatticeMaze`、`SolvedMaze`、`TargetedLatticeMaze`；连接关系、像素栅格、ASCII 等 |
| **`maze_dataset/generation/`** | 迷宫生成算法注册与实现（如 `LatticeMazeGenerators.gen_dfs`） |
| **`maze_dataset/dataset/`** | `MazeDataset`、`MazeDatasetConfig`：按配置批量生成/加载数据集；过滤器、集合类 |
| **`maze_dataset/tokenization/`** | 将迷宫转为 token 序列（新旧 tokenizer、模块化系统等），供序列模型训练 |
| **`maze_dataset/plotting/`** | 迷宫与 token 的可视化（如 `plot_maze`、`plot_tokens`） |
| **`maze_dataset/benchmark/`** | 性能与配置扫描等基准脚本 |
| **`notebooks/`** | 教程型 notebook（生成器、数据集、tokenization 等） |
| **`tests/`** | 单元测试 |
| **`docs/`** | 文档站点资源与论文材料 |

**数据流直觉**：`generation` 造迷宫 →（可选）求解得到 `SolvedMaze` → `dataset` 打包成 `MazeDataset` → `tokenization` 或 `as_pixels` 等变成 **训练/评测输入**。

---

## 三、环境与安装（必读）

### 3.1 Python 版本

官方在 `pyproject.toml` 中声明：**`requires-python = ">=3.10"`**。  
若你当前 **VoCoT conda 环境是 Python 3.9**，可能仍能装部分依赖，但与上游测试矩阵不一致，**建议为 maze-dataset 单独建 3.10+ 环境**，或在 VoCoT 环境中升级到 3.10+ 后再试。

### 3.2 Python 3.9 与 `TypeError: unsupported operand type(s) for |: 'MetaAbstractArray'...`

在 **Python 3.9** 下，若某处注解写成 `Int[np.ndarray, "..."] | Int[np.ndarray, "..."]`，**jaxtyping** 可能在 **import 时求值**并触发上述错误。本机 **maze-dataset** 已在 `maze_dataset/utils.py` 顶部加入 **`from __future__ import annotations`**，以**推迟注解求值**，从而避免该问题。

若你使用 **官方未打补丁的 releases**，更稳妥的做法仍是：**使用 Python 3.10+**，或在 maze-dataset 目录执行 `pip install -e .` 后确认环境与 `pyproject.toml` 一致。

### 3.3 常见报错：`ModuleNotFoundError: No module named 'jaxtyping'`

**原因**：`maze_dataset` 在 import 时会拉入 `jaxtyping` 等依赖；未安装则报错。

**推荐做法**（在 **maze-dataset 仓库根目录**）：

```bash
cd /path/to/maze-dataset
pip install -e .
```

这会根据 `pyproject.toml` 安装 **jaxtyping、muutils、zanj、matplotlib、numpy、tqdm** 等。

**最小补包**（仅应急，仍建议 `pip install -e .`）：

```bash
pip install "jaxtyping>=0.2.19" "muutils>=0.8.3" "zanj>=0.5.0" matplotlib tqdm
```

### 3.4 `ImportError: cannot import name '_FORMAT_KEY' from 'muutils...'`

**原因**：较新的 **muutils** 把 **`_FORMAT_KEY`** 从 **`muutils.json_serialize.util`** 挪到了 **`muutils.json_serialize.types`**，而官方 maze-dataset 部分文件仍从 `util` 导入，导致 `ImportError`。

**处理**：

- **推荐**：使用 **Python 3.10+**，并在 maze-dataset 目录执行 **`pip install -e .`**，保证 `pyproject.toml` 锁定的一组版本。
- **本仓库旁的 maze-dataset**：已增加 **`maze_dataset/compat_muutils.py`**，统一从正确位置解析 `_FORMAT_KEY` 与 `JSONdict`，可与新版 muutils 共存。

### 3.5 为何装了 muutils 仍不能用 Python 3.9？

maze-dataset 声明 **`requires-python >= 3.10`** 是有原因的：例如 **`muutils.serializable_dataclass`** 在 **Python 3.9** 下对 **`kw_only`** 会直接报错（`KWOnlyError`），而 **`LatticeMaze`** 等类依赖该机制。**VoCoT 的 `run_maze_inference_demo.py` 会在导入前检查版本**，若 `< 3.10` 会给出创建 3.10 环境的说明。

**结论**：跑迷宫 demo 请使用 **Python 3.10+** 的解释器，而不是强改 3.9。

### 3.6 其它可选能力

- **完整 tokenization 额外依赖**：见 `pyproject.toml` 中 `[project.optional-dependencies]`（如 `frozendict`、`rust_fst` 等）。
- **开发/文档**：可用 `uv` 同步 `dependency-groups`（见上游 README / makefile）。

---

## 四、VoCoT：`run_maze_inference_demo.py` 快速开始

### 4.1 脚本做什么？

- 默认在 **`VoCoT` 上一级目录** 查找 **`maze-dataset`**（即 `../maze-dataset`），用 **`MazeDataset`** 现场生成若干迷宫；
- 为每条样本保存 **两张图**（在自动生成模式下）：
  - **`maze_XXXX.png`**：**无解救路径**，绿点起点、红点终点——**送给多模态模型**做 VQA；
  - **`maze_XXXX_with_path.png`**：**含解救路径高亮**——给人看 **标准路径与结果**；
- 生成简单二分类题（起点是否在终点「左侧」）及 **yes/no** 标签；
- 加载 **VoCoT / Volcano** 模型推理，将 **`manifest.json` / `results.json` / 图片副本** 写到 `output/maze/<模型名>/inference_demo/`。

### 4.2 一键命令（在 VoCoT 仓库根目录）

```bash
cd /home1/cjl/MM_2026/VoCoT

# 依赖已装好的前提下
python run_maze_inference_demo.py

# 调整数量、迷宫边长
python run_maze_inference_demo.py --n 8 --grid_n 9 --seed 0

# 不要带路径的参考图（仅省磁盘）
python run_maze_inference_demo.py --no_solution_viz

# 使用已有 JSON + 图片目录（需同时指定两者）
python run_maze_inference_demo.py --questions_json /path/to/maze_eval_samples.json --image_dir /path/to/images
```

### 4.3 输出在哪里？

默认：

```text
output/maze/<本地模型目录名>/inference_demo/
  manifest.json
  results.json
  images/                    # 推理用图的副本（及带路径图副本，若存在）
  generated/                 # 自动生成时：原始 maze_eval_samples.json 与源图
    images/
      maze_0000.png
      maze_0000_with_path.png
      ...
    maze_eval_samples.json
```

对照 **`answer`** 与模型 **`prediction`**，并用 **`*_with_path.png`** 核对迷宫路径是否合理。

### 4.4 多模态模型权重（VoCoT）

本脚本与 `run_inference_demo.py` 一致：默认使用 **`project.py`** 中的 **`volcano_7b_luoruipu1_path`**（若目录存在），否则 Hugging Face id **`luoruipu1/Volcano-7b`**。无需单独下载 maze 专用小模型；需本地 Volcano 权重与 CLIP 等按 VoCoT **README** 准备。

---

## 五、训练（可选，上游能力）

**maze-dataset 本身**侧重 **数据生成与处理**；端到端 **训练迷宫序列模型** 通常配合 ** [maze-transformer](https://github.com/understanding-search/maze-transformer)** 等上游项目完成。若仅需在 VoCoT 里 **试推理**，只需装好 maze-dataset + Volcano 推理环境即可，**不必**跑 maze-dataset 的训练流程。

若要深入训练，请查阅：

- maze-dataset **README** 与 **notebooks/**
- maze-transformer 仓库说明与超参

---

## 六、小结

| 事项 | 建议 |
|------|------|
| **用途** | 程序化迷宫数据：生成、求解、过滤、可视化、序列化 |
| **Python** | 优先 **3.10+** |
| **报错 jaxtyping** | 在 maze-dataset 目录 `pip install -e .` |
| **看路径** | 自动生成时使用 **`maze_XXXX_with_path.png`**；结果目录见 `output/maze/...` |
| **VoCoT 推理** | `python run_maze_inference_demo.py`（本仓库根目录） |

以上内容与上游版本可能随 maze-dataset 更新而变化；以你磁盘上的 **`pyproject.toml`** 与 **官方文档** 为准。
