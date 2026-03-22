# GQA 评测说明（VoCoT）

本文梳理在本仓库中跑 **GQA** 评测时的流程、指标、数据格式、产出物与规模；默认以 `run_gqa_bench.py` 与数据集根目录 `project.gqa_bench_path`（一般为 `data_path/GQA_Bench`）为准。

---

## 1. 整体流程

```
run_gqa_bench.py
  → eval/evaluate_benchmark.py（VoCoT 推理，CoT）
  → eval/eval_tools/convert_res_to_gqa.py（格式转换）
  → [仅 val] 数据集自带 GQA_Bench/eval/eval.py（官方指标）
```

- **val**：可对 **balanced val** 问题集在本地算出准确率等（需完整预测文件）。
- **testdev**：无公开标准答案；脚本只生成与官方一致的 **predictions JSON**，用于后续提交或自行对齐流程。

---

## 2. 官方 `eval.py` 计算的指标（GQA 论文配套）

脚本路径：`GQA_Bench/eval/eval.py`。在 **balanced** 子集上汇总的主要指标包括（见脚本内 `metrics` / `detailedMetrics`）：

| 指标 | 含义（简述） |
|------|----------------|
| **Accuracy** | 预测答案与标注答案是否一致（balanced 上平均，百分比）。 |
| **Binary / Open** | 二分类式问题 vs 开放式 query 的准确率子集。 |
| **Consistency** | 模型在「蕴含题」上的一致性（需对 **val 全量** 等问题集提供预测，且需加 `--consistency`）。 |
| **Validity** | 答案是否在题目允许的合法答案集合内（来自 `val_choices.json`）。 |
| **Plausibility** | 答案是否「合理」（场景图推导的合理答案集合）。 |
| **Grounding** | 注意力/区域是否对准相关物体（需提供 `--attentions` 且 `--grounding`，本仓库默认评测 **不启用**）。 |
| **Distribution** | 预测与真实答案在全局分组上的分布差异（chi-square 相关，**越低越好**）。 |

**细分报表**（同一脚本会打印）：

- Accuracy / **structural type**（逻辑结构类型）
- Accuracy / **semantic type**（语义类型）
- Accuracy / **reasoning steps**（推理步数）
- Accuracy / **question length**（问句长度）

### 2.1 `run_gqa_bench.py` 默认实际会打印哪些

当前对官方 `eval.py` 的调用**未**传 `--consistency`、`--grounding`，因此：

- 会输出：**Binary、Open、Accuracy、Validity、Plausibility、Distribution**，以及上述 **四类细分 Accuracy**。
- **不会**计算并打印 **Consistency**、**Grounding**（脚本里会提示可考虑加这些选项）。

若需要 **Consistency**，需按官方说明对更大问题集生成预测并加上 `--consistency`（见 `eval.py` 开头注释与参数）。

---

## 3. VoCoT 侧推理在评什么

- **任务**：视觉问答；开启 **`--cot`** 时为两阶段：先 CoT，再在「What is your final answer?」轮输出 **最终短答案**。
- **写入结果的字段**：每条样本含 `item_id`（形如 `eval_<下标>`）、**`prediction`**（最终答案字符串）；CoT 时另有 **`thought`**。
- **与 GQA 官方对齐**：`convert_res_to_gqa.py` 用 **`item_id` 下标** 对应 `questions.json` 字典的 **key 顺序**（与 `GQADataset` 遍历顺序一致），得到 `questionId`，并把 `prediction` 做小写、去 `</s>`、去句末 `.` 等规范化。

---

## 4. 输入 / 输出格式

### 4.1 模型与数据输入（概念上）

- **图像**：`base_path/{imageId}.jpg`（`base_path` 一般为 `.../GQA_Bench/images/images`）。
- **问题**：来自 `questions1.2/*.json` 中每条记录的 `question` 等字段；评测类为 `GQADataset`（`locals/datasets/eval/short_qa.py`）。

### 4.2 `evaluate_benchmark.py` 产出（中间结果）

- **路径**：`{output_dir}/{数据集 yaml 主文件名}.json`  
  例如：`output/gqa/<模型名>/cot/GQA_bench_val.json`
- **格式**：JSON **数组**，每项大致为：

```json
{
  "item_id": "eval_0",
  "thought": "...",
  "prediction": "yes"
}
```

（无 CoT 时无 `thought`；与 `evaluate_loss` 模式字段不同，GQA 默认走 **生成** 分支。）

### 4.3 `convert_res_to_gqa.py` 产出（交给官方 `eval.py`）

- **路径**：val 为 `val_balanced_predictions.json`，testdev 为 `testdev_balanced_predictions.json`（均在 `output_dir` 下）。
- **格式**：JSON **数组**，官方约定：

```json
[
  {"questionId": "00123456", "prediction": "cat"},
  ...
]
```

`questionId` 为 GQA 题目 ID 字符串；`prediction` 为规范化后的短文本。

### 4.4 官方 `eval.py` 的输入文件（val）

由 `run_gqa_bench.py` 传入（均基于 `gqa_root`）：

| 参数 | 典型路径 |
|------|-----------|
| `--scenes` | `sceneGraphs/val_sceneGraphs.json` |
| `--questions` | `questions1.2/val_balanced_questions.json` |
| `--choices` | `eval/val_choices.json` |
| `--predictions` | 上一步生成的 `val_balanced_predictions.json` |

---

## 5. 预期结果应如何理解

- **val**：终端会打印各 **百分比** 指标及细分；**Accuracy** 为最常用的主指标之一；**Distribution** 标注为 *lower is better*。
- **testdev**：本地 **不跑** `eval.py` 打分；仅确认生成了完整的 `testdev_balanced_predictions.json`，用于榜单或后续流程。
- **无固定「及格分」**：与模型规模、是否 CoT、是否对齐训练分布有关；需与论文/其它实现对比。

---

## 6. 评测数据量（当前数据集 JSON 条数）

以下为本机 `questions1.2` 下 **balanced** 问题字典 **key 数量**（每条 key 一题，以你磁盘上的文件为准）：

| Split | 文件 | 题目条数（约） |
|-------|------|----------------|
| val | `val_balanced_questions.json` | **132062** |
| testdev | `testdev_balanced_questions.json` | **12578** |

图像张数由 `imageId` 去重决定，与题目条数不同；图片根目录为 `images/images/`。

---

## 7. 评测时间（量级说明）

无法给出统一「秒数」，与以下强相关：

- GPU 型号与数量、`--nproc`（多卡并行）、每题 **CoT + 二次生成**、`max_new_tokens=2048`；
- `evaluate_benchmark` 为 **batch_size=1** 顺序推理；
- 官方 `eval.py` 在 CPU 上遍历预测与场景图，相对推理通常 **占比很小**。

**经验上**：全量 val（约 13 万题）单次推理往往是 **数小时级到数十小时级**（视 GPU 而定）；testdev（约 1.26 万题）同比例缩短。正式跑之前建议缩短流程做冒烟测试（例如临时改用小问题子集或单独写脚本只跑前 N 条；**注意** `evaluate_benchmark` 的 `--sub_sample` 对部分数据集是切片 `meta` 列表，与 GQA 的字典结构未必兼容，需自行验证）。

---

## 8. 相关入口与配置

| 说明 | 路径 |
|------|------|
| 一键评测入口 | `run_gqa_bench.py`（仓库根目录） |
| 推理入口 | `eval/evaluate_benchmark.py` |
| 格式转换 | `eval/eval_tools/convert_res_to_gqa.py` |
| 数据集类 | `locals/datasets/eval/short_qa.py` → `GQADataset` |
| 路径常量 | `project.py`：`gqa_bench_path`、`volcano_7b_luoruipu1_path`、本地 CLIP `clip_vit_large_patch14_336_path` |
| 官方评测脚本 | `GQA_Bench/eval/eval.py` |

更多背景与下载说明见 [GQA 官网](https://gqadataset.org/)。
