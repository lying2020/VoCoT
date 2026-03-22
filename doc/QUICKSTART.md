# VoCoT 本地快速开始

面向在本仓库根目录下开发与运行的说明；与官方 [README.md](../README.md) 互补，补充环境细节、目录说明与一键推理命令。

**多轮对话 / 多 HOP 文档式推理**（调测脚本与 API）见 [MULTIHOP.md](./MULTIHOP.md)。

**保存推理结果、框可视化、RefBind 说明**：使用 `run_inference_save.py`（结果写入 `output/`，并复制 [REFBIND_AND_DIMENSIONS.md](./REFBIND_AND_DIMENSIONS.md)）。

---

## 1. Conda 环境与依赖

仓库未提供 `environment.yml`，以 **`requirements.txt`** 为准。README 建议使用 **Python 3.9**。

### 一键脚本（推荐）

在仓库根目录执行：

```bash
cd /path/to/VoCoT
./setup_conda.sh cu121    # 先装 CUDA 12.1 对应的 torch/torchvision，再装 requirements
# 或：./setup_conda.sh cpu
# 或：TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 ./setup_conda.sh
```

环境名默认为 `vocot`，可通过 `CONDA_ENV_NAME=myenv ./setup_conda.sh` 修改。若已存在同名环境，脚本会跳过创建并仍执行 `pip install`；若要删掉重建：`FORCE_RECREATE=1 ./setup_conda.sh cu121`。

### 手动步骤

```bash
cd VoCoT  # 本仓库根目录

conda create -n vocot python=3.9 -y
conda activate vocot
pip install --upgrade pip
```

**PyTorch：** 请按本机 CUDA 版本在 [PyTorch 官网](https://pytorch.org) 选择安装命令，先安装 `torch` / `torchvision`，再安装其余依赖，可减少版本冲突。

```bash
pip install -r requirements.txt
```

**Flash Attention（可选，用于加速）：**

```bash
pip install flash-attn --no-build-isolation
```

若编译失败可暂时跳过。部分环境需与 CUDA、编译器版本匹配。

**系统包（README 中的 GUI 依赖，无桌面可跳过）：**

```bash
sudo apt-get install python3-tk -y
```

**常见注意：**

- 保持 `requirements.txt` 中的 `transformers==4.37.2` 等锁定；`diffusers` 已调整为 `>=0.30,<0.36`（与新版 `huggingface_hub`、`accelerate` 兼容，见下「常见问题」）。
- 核心栈：`torch>=2.0`、`transformers`、`accelerate`、`peft`、`lightning`、`timm`、`open_clip_torch`、`xformers`、`deepspeed` 等。

### 常见问题：NumPy / matplotlib / diffusers

| 现象 | 处理 |
|------|------|
| `AttributeError: _ARRAY_API` / `numpy.core.multiarray failed to import`（NumPy 2 + 旧 matplotlib） | 使用 NumPy 2 时安装 **`matplotlib>=3.8`**：`pip install "matplotlib>=3.8.0,<4"`；或改用 **`numpy<2`** 并搭配 **`matplotlib==3.7`**。 |
| `cannot import name 'cached_download' from 'huggingface_hub'`（旧 diffusers 0.14） | 升级 **`diffusers>=0.30`**：`pip install "diffusers>=0.30.0,<0.36.0"`，仓库已兼容 `DiagonalGaussianDistribution` 的新旧导入路径。 |
| 推理时仍请求 `huggingface.co/openai/clip-vit-large-patch14-336` | 将 CLIP 按 HuggingFace 格式放到 **`project.py` 里的 `clip_vit_large_patch14_336_path`** 指向的目录；或设置环境变量 **`VOCOT_LOCAL_CLIP_PATH=/你的/clip-vit-large-patch14-336`**。`load_model` 会把配置里的 `openai/clip-*` 自动替换为该本地路径。 |

---

## 2. 下载模型权重

论文中的 **VolCano-7B** 发布在 Hugging Face：

- 模型：[luoruipu1/Volcano-7b](https://huggingface.co/luoruipu1/Volcano-7b)

可将权重下载到本地目录，推理时用 `--model_path` 指向该目录；若已配置网络与 Hugging Face 访问，也可直接使用上述 Hub id 作为路径（由 `transformers` 自动下载）。

**说明：** 本仓库的 VolCano（VoCoT）与 KAIST 另一项目中的 Volcano 命名相近但权重与代码基不同；请使用 **`luoruipu1/Volcano-7b`** 配合本仓库代码。

---

## 3. 推理（最简）

### 方式 A：脚本（推荐）

在仓库根目录执行：

```bash
conda activate vocot
# 不传 --model_path 时：若 project.py 中 volcano_7b_luoruipu1_path 目录存在则自动使用该路径，否则用 Hugging Face luoruipu1/Volcano-7b
python run_inference_demo.py \
  --image figs/sample_input.jpg \
  --query "Describe the image." \
  --precision fp16 \
  --device cuda:0
```

关闭 VoCoT 链式/grounding 提示（不附加 `COT_ACTIVATION` 类后缀）：

```bash
python run_inference_demo.py \
  --image figs/sample_input.jpg \
  --query "What is in the image?" \
  --no_cot
```

### 方式 B：与 README 一致的 Python 片段

```python
from model.load_model import load_model, infer
from PIL import Image

model, preprocessor = load_model("luoruipu1/Volcano-7b", precision="fp16")
image = Image.open("figs/sample_input.jpg").convert("RGB")
response = infer(model, preprocessor, image, "Describe the image.", cot=True)
print(response[0])
```

### 输入 / 输出约定

| 项目 | 说明 |
|------|------|
| **输入图像** | RGB 图像；默认预处理会 **expand to square**（与 README 一致）。 |
| **输入文本** | 普通问题字符串；`cot=True` 时内部会拼接 `<ImageHere>`、`<grounding>` 与 `constants.py` 中的 `COT_ACTIVATION`。 |
| **输出** | `infer` 返回 `list[str]`，取 `[0]` 为当前 batch 的解码文本；可能含推理过程与 `<coor>…</coor>` 归一化框（相对**扩方后的图像**坐标系）。 |

---

## 4. 仓库结构（关键路径）

| 路径 | 作用 |
|------|------|
| `model/` | VolCano：语言模型（Mistral）、视觉编码器（CLIP 等）、投影与可选生成模块 |
| `locals/datasets/` | 数据集与 `VoCoT_InputProcessor`、`SFT_DataCollator` |
| `config/experiments/` | 分阶段训练配置（如 `stage1_alignment.yaml`） |
| `config/datasets/` | 训练/评测数据组合 yaml |
| `config/deepspeed/` | DeepSpeed 配置 |
| `train_volcano.py` | 训练主入口（`torchrun` + 实验 yaml） |
| `eval/` | 评测说明与脚本，见 `eval/Evaluation.md` |
| `constants.py` | 图像占位、坐标与 COT 相关常量 |
| `run_inference_demo.py`（仓库根目录） | 命令行一键推理示例；默认模型见 `project.volcano_7b_luoruipu1_path` |

---

## 5. 数据与训练（可选）

- **VoCoT-Instruct 等数据：** 见 README [Data](README.md#data) 与 [Hugging Face 数据集](https://huggingface.co/datasets/luoruipu1/VoCoT)；图像需从 GQA、COCO、LVIS 等来源自行准备。
- **训练命令示例（需先修改 yaml 内路径）：**

```bash
torchrun --nproc_per_node=8 train_volcano.py --conf config/experiments/stage1_alignment.yaml
```

`nproc_per_node` 需不大于本机 GPU 数量；单机少卡时需相应调整 batch 与梯度累积等。

---

## 6. 模型与算力（摘要）

- **结构：** Mistral-7B-Instruct 系语言模型 + CLIP ViT-L/14（336 输入）等；详见 `config/experiments/stage1_alignment.yaml` 与论文。
- **仅推理：** 使用发布权重即可，无需训练。
- **训练：** 多阶段、DeepSpeed、多卡更现实；7B 多模态训练通常需 **多卡与较大显存**（视是否 LoRA、梯度检查点等而定）。
- **评测：** 见 `eval/Evaluation.md` 与 `eval/commands/test_all_benchmark.sh`。

---

## 7. 多轮对话（进阶）

单次 `infer()` 默认只构造一轮 `human` 消息。若需在同一图像上多轮追问，可将历史 `human`/`gpt` 轮次按 `locals/datasets/preprocessor.py` 中 `preprocess_llama_2` 的约定拼成 `conversation` 列表，并仍传入 `input_images`；具体见代码与对话模板（Mistral instruct 格式）。

---

若环境与 CUDA 版本固定，可将「官方 PyTorch 安装命令」补记在本文件末尾，便于实验室复现。
