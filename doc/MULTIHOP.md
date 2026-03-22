# 多轮与多 HOP 推理调测说明

VoCoT 默认单轮 `infer()` 会在一轮内做带 grounding 的链式推理。若需要 **多轮对话** 或 **单次输出多节「文档」式多跳**，可用下面两种方式。

---

## 1. 单次生成：多 HOP 文档（推荐先做）

在一次 forward 里要求模型按小节输出，适合 **GQA 式多跳**、可读的 Markdown 结构。

**API：** `model.load_model.infer_multihop_document`  
**默认**在问题后附加 `constants.MULTIHOP_DOC_INSTRUCTION`（可改该常量或传 `hop_instruction=`）。

**命令行：**

```bash
cd VoCoT
python run_multihop_demo.py --mode document \
  --query "分析图中关系，并分多跳给出结论。" \
  --max_new_tokens 2048
```

**自定义多跳说明（覆盖默认英文 instruction）：**

```bash
python run_multihop_demo.py --mode document --query "..." \
  --hop_instruction_file my_hop_prompt.txt
```

**调测要点：**

- `max_new_tokens` 建议 **≥ 1024**，多跳时 **2048** 更稳。
- 若输出未按 `### HOP` 分节，可在 `constants.py` 里加强 `MULTIHOP_DOC_INSTRUCTION` 的格式约束（或换成中文指令）。
- 仍受 **单次生成长度** 限制；极长文档需分段多次请求或后处理。

---

## 2. 多轮对话：同一图像 + 历史 + 最后一问

把 **user / assistant** 交替轮次（含历史）拼进模板，**只生成最后一轮 assistant**（与训练时 `preprocess_llama_2` + `inference=True` 一致）。

**API：** `infer_multiturn(model, preprocessor, image, turns, cot_last=True, ...)`  
`turns`：`[('user', '...'), ('assistant', '...'), ..., ('user', '最后一问')]`。

**JSON 示例：** `examples/multiturn_dialog.json`

```bash
python run_multihop_demo.py --mode multiturn \
  --dialog examples/multiturn_dialog.json \
  --print_turns
```

**调测流程建议：**

1. **第一轮** 用 `run_inference_demo.py` 得到 `assistant_1`。
2. 将 `assistant_1` 写入 JSON 的第二条，再写 **第二轮 user**。
3. 再跑 `run_multihop_demo.py --mode multiturn --dialog your.json`。

**注意：**

- 模型主要在 **单轮 VoCoT 数据** 上训练，**多轮** 为模板扩展，稳定性需自行评测。
- 仅 **首轮 user** 带 `<ImageHere>` + `<grounding>`；**仅最后一轮 user** 在 `cot_last=True` 时附加 `COT_ACTIVATION`。
- 历史 `assistant` 建议用**真实模型上轮输出**，不要用占位句糊弄，否则分布偏移更大。

---

## 3. Python 调用片段

```python
from model.load_model import load_model, infer_multihop_document, infer_multiturn
from PIL import Image

model, preprocessor = load_model("/path/to/Volcano-7b", precision="fp16")
image = Image.open("images/sample_input.jpg").convert("RGB")

# 多 HOP 文档（单次）
out = infer_multihop_document(
    model, preprocessor, image,
    "图中是否存在 A 在 B 左侧的关系？请分跳推理。",
    cot=True,
    max_new_tokens=2048,
)
print(out[0])

# 多轮
turns = [
    ("user", "图中主要物体有哪些？"),
    ("assistant", "上一轮模型的回答..."),
    ("user", "这些物体之间的空间关系？"),
]
out = infer_multiturn(model, preprocessor, image, turns, cot_last=True, max_new_tokens=2048)
print(out[0])
```

---

## 4. 与「真多轮交互」的区别

此处 **multiturn** 仍是 **一次** `generate`：上下文里包含历史 token，**不**在脚本里循环调用多次 `infer`。若要做「每轮单独调用、再拼上下文」，需要在外层循环里维护 `turns` 并每次追加新 assistant，再调 `infer_multiturn`（逻辑等价，便于插入人工或工具反馈）。

---

## 5. 评估多跳质量（可选）

- 看 **HOP 节** 是否覆盖问题、**Final Answer** 是否简洁。
- 若有 `<coor>`，可用 `utils/eval_util.py` 中解析函数与 GT 框对比（需自建标注）。
- 自动化指标可借鉴 GQA、VSR 等多跳任务的数据格式，自行写脚本比对最终答案字符串。
