# RefBind / `</coor>` 绑定：代码位置与算法理解（含「多 patch → 多 token」）

本文说明 **VoCoT 仓库**里与 **RefBind（`sub_image_bind=True`）**、**整图 box 对齐（`sub_image_bind=False`）** 相关的实现，以及 **多个 patch 如何变成接在生成序列后面的多段 embedding**。

---

## 一、代码位置速查（建议按此顺序阅读）


| 主题                | 文件                                                                | 大致行号                                            | 说明                                                |
| ----------------- | ----------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------- |
| 推理入口是否 RefBind    | `model/load_model.py`                                             | `sub_image_bind = False`                        | 默认 **关闭** RefBind                                 |
| 生成时是否走绑定          | `model/language_model/volcano_mistral.py`                         | `prepare_inputs_for_generation` ~374–417        | 检测 **最后一个生成 token** 是否为 `</coor>`（`eoc_token_id`） |
| 整图框对齐（非 RefBind）  | 同上                                                                | `generate_box` ~420–433                         | `sub_image_bind=False` 时调用                        |
| 子图 RefBind        | 同上                                                                | `generate_sub_image` ~435–456                   | `sub_image_bind=True` 时调用                         |
| 框内多 patch 抽取      | `model/language_model/volcano_llama.py`（`VolCanoMetaForCausalLM`） | `box_align` ~308–323                            | **核心几何**：在 **2D patch 网格**上裁矩形，展平为 **多 token**    |
| 整图编码（得到 patch 序列） | 同上                                                                | `encode_img` ~325+                              | 视觉塔 + `front_mm_projector`，首路输出供后续对齐              |
| 训练时多框、变长 token    | 同上                                                                | `prepare_inputs_labels_for_multimodal` ~444–453 | `box_align` 与 **变长** `box_feat_len` 的拼接逻辑         |
| 评测里打开子图           | `eval/evaluate_benchmark.py`                                      | `--sub_image`、`model.sub_image_bind`            | 与训练/评测一致时启用 RefBind                               |


> 说明：`VolCanoMistralForCausalLM` 继承 `VolCanoMetaForCausalLM`，`generate_box` / `box_align` / `encode_img` 的实现在 **llama/base** 一侧；`volcano_mistral.py` 里 **重写或挂载** 了生成钩子（`prepare_inputs_for_generation`）。

---

## 二、「多个 patch」如何接到生成 token 后面？

### 1. 触发时机（自回归生成）

在 **Hugging Face `generate`** 的每一步，`prepare_inputs_for_generation` 会看 **当前步刚生成的最后一个 token**：

- 若等于 `**</coor>`**（`eoc_token_id`），且 `**no_bind` 为 False**，则 **不**再只喂下一个词表 id，而是：
  1. 从 **已生成序列**里，从 `<coor>`（`boc_token_id`）起 **decode 出当前框的文本**；
  2. 用 `extract_box_str(..., mistral=True)` 得到 **4 个归一化浮点数**（相对 **扩方后** 图像，与训练一致）；
  3. 调用 `**generate_box`** 或 `**generate_sub_image**`，得到一段 `**inputs_embeds**`；
  4. 将这段 embedding **接在**「当前步已生成的 token（含 `</coor>`）」的 embedding 后面，**继续** forward。

对应代码：`volcano_mistral.py` 中 `new_token_ids == self.eoc_token_id and not self.no_bind` 分支。

**注意：** 这里接在后面的是 **连续向量（embedding）**，不是词表里叫 “image token” 的离散 id；长度由下面两种路径决定。

### 2. 路径 A：`sub_image_bind=False` → `generate_box` → `box_align`（整图特征网格）

**输入：**

- `cache_images`：由 `**encode_img(整图)`** 得到的第一路特征（实现里期望能看成 `**[num_patches, hidden]**`，且 `num_patches` 为 **完全平方数**，以便 `reshape` 成 `sqrt×sqrt` 网格）。  
  - 缓存：`generate_box` 里若 `cache_images is None`，则 `self.cache_images = encode_img(...)[0]`。
- `current_box`：**形状 `(4,)`**，归一化到 **[0,1]**，经 `num_patches * box` 映射到 **patch 下标范围**（见 `box_align` 里 `bboxes_index = num_patches * bboxes`）。

`**box_align` 在做什么（`volcano_llama.py`）：**

- 把一维特征 `image` 视为 `**[num_patches, hidden]`**，再 `reshape(num_patches, num_patches, hidden)`。
- 对每个框，用 **floor/ceil** 得到在网格上的 **矩形区间** `[x_min:x_max, y_min:y_max]`（**注意**：此处索引语义与 `image_feat_2d` 的切片写法需结合源码理解，本质是 **框内所有 patch 单元**）。
- 将该矩形内所有 patch 的向量 **展平** 为 `**[N, hidden]`**，其中 `**N = 框内 patch 个数**`，**随框大小变化**。
- 返回 `box_feat` 的列表；`generate_box` 取 **当前框** 的 `**box_feat[0]`**，形状 `**[N, hidden]**`。

**接到序列上：**

```text
next_inputs_embeds = cat( embed(当前 input_ids 这一段), box_feat, dim=序列维 )
query_len = box_feat.shape[0]   # 即 N，可变
```

`prepare_inputs_for_generation` 里用 `**query_len**` 扩展 `attention_mask`（`torch.ones(1, query_len)`）。

**结论：**  

- **「一个框」对应 **N 个** 连续视觉 token（每个对应一个 patch 的特征），**不是**固定 1 个 token。  
- **N** 由框在 **patch 网格**上覆盖的格子数决定；框越大，**接在 `</coor>` 后面的 embedding 越长**。

### 3. 路径 B：`sub_image_bind=True`（RefBind）→ `generate_sub_image`

**输入：**

- `cache_raw_image`：**未扩方**的 PIL，在 `condition_completion` 里设为 `input_dict['raw_images'][0][0]`。
- 同样从序列里解析出 **4 个归一化数**，映射到 **像素** 后 `**crop`** 子图，再 `**resize_image_to_square**` → 与训练一致的 CLIP 预处理 → `**encode_img(sub_image)**`。

**输出：**

- `box_feat = encode_img(sub_image)[0][0]`：这里是 **整张子图再走一遍「全局图像编码」**，长度通常是 **固定的 `num_query_token`（如 Q-Former query 数）或 `n_query`**，与「框内包含多少个原图 patch」**无直接一一对应**，而是 **一整张子图的紧凑表示**。

**接到序列上：** 同样是 `cat(..., box_feat, dim=1)`，`query_len = box_feat.shape[0]`（一般 **固定**，由架构决定）。

**结论：**  

- RefBind 路径是 **「裁子图 → 当一整张新图编码」**，多 patch **被压在** 视觉塔 + projector 的 **固定长度** 输出里；  
- 与路径 A **显式展开框内每个 patch** 不同。

---

## 三、如何理解整个「算法」

1. **语言模型先**在词表里生成 `**<coor> 四个数 </coor>`** 这一段 **离散 token**。
2. 当 **生成到 `</coor>`** 时，钩子 **不进入下一词表 id**，而是 **注入一段视觉 embedding**：
  - **非 RefBind**：从 **整图** 的 patch 特征网格里 **抠出框内所有 patch**，**按顺序拼成 N 个向量**，接在后面；
  - **RefBind**：用框 **crop 原图**，把 crop **当成新图**编码成 **固定长度** 向量序列，接在后面。
3. 之后的 **LM 层**把这些向量当作 **额外的「位置」**，继续自回归生成自然语言。

**输入/输出小结：**


| 项目                | 非 RefBind（`generate_box`）          | RefBind（`generate_sub_image`） |
| ----------------- | ---------------------------------- | ----------------------------- |
| 几何输入              | 整图 tensor + 归一化框                   | 原图 PIL + 归一化框                 |
| 视觉来源              | 整图 `encode_img` 缓存的 **patch 网格特征** | **crop 子图** 再 `encode_img`    |
| 接在 `</coor>` 后的长度 | **可变 N**（框内 patch 数）               | 通常 **固定**（子图编码输出长度）           |
| 默认 `load_model`   | ✅ 使用该路径                            | ❌ 未启用                         |


---

## 四、与 `n_query` / `<Img>` 生图分支的区别

- `**n_query`**：多用于 **整张图** 首次插入时的 **query 数**（如 Q-Former）或 **扩方图上的 patch 数**定义，和 `**<Img>` 触发 `generate_image`** 的分支相关（`volcano_mistral.py` 里 `boi_token` 分支），与 `**</coor>` 的 `box_align` 变长 N** 是 **不同概念**。
- 若只关心 `**</coor>` 后接了多少维**：以 `**box_feat.shape[0]`**（或日志里扩展的 `attention_mask` 长度）为准。

---

## 五、延伸阅读

- 坐标与图像预处理：`README.md`（expand to square）、`doc/REFBIND_AND_DIMENSIONS.md`  
- 可视化脚本：`utils/vocot_output_viz.py`、`run_inference_save.py`

