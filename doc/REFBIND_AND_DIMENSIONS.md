# VoCoT / VolCano：RefBind（子图绑定）与 `</coor>` 绑定的维度说明

本说明对应 `model/language_model/volcano_mistral.py` 中生成阶段逻辑。

## 1. 如何判断「是否用到 RefBind」

代码里与论文/界面常说的 **RefBind** 对应的是 **`sub_image_bind=True`** 分支：

- **`sub_image_bind=False`（`load_model` 默认）**  
  当模型在生成中写出 **`</coor>`**（`eoc_token`）时，走 **`generate_box()`**：  
  用当前已解码的 `<coor>…</coor>` 里的 **四个归一化数**（相对 **扩方后** 整图），在 **整图视觉特征** 上做 `box_align`，得到一段 **视觉向量**，再 **拼到序列后面** 继续生成。  
  **不**对原图做 crop 再编码。

- **`sub_image_bind=True`**  
  同上触发条件（生成到 `</coor>`），但走 **`generate_sub_image()`**：  
  用 **未扩方的 PIL 原图** `cache_raw_image`，按同一组归一化坐标 **裁子图** → `resize_image_to_square` → 视觉塔 **encode 子图** → 向量拼到序列后。  
  这路需要 `condition_completion` 里提供 **`raw_images`**（见 preprocessor），评测里常通过 `--sub_image` 打开。

**结论：** 默认 **`run_inference_demo.py` / `infer()` 未开启 RefBind**；只要 **`model.sub_image_bind` 为 False**，即使用到「坐标绑定」，也是 **整图特征上的 box 对齐**，不是子图 RefBind。

保存结果里的 **`meta.json`** 会写明本次运行的 **`sub_image_bind`** 与 **`refbind_used`**（与 `sub_image_bind` 一致）。

## 2. `</coor>` 触发时，张量大致怎么流（`sub_image_bind=False`）

以下在 **batch=1** 下描述。

| 步骤 | 含义 | 典型形状（示意） |
|------|------|-------------------|
| 整图输入 | 扩方后的图经 CLIP 预处理，`encode_img` | 与视觉塔输出相关，内部为 patch 特征 |
| `extract_box_str(..., mistral=True)` | 从 `<coor>` 起解码到 token 序列，得到 4 个 **[0,1]** 浮点 | `(4,)` |
| `current_box.unsqueeze(0)` | 归一化框 | `(1, 4)` |
| `box_align(cache_images[0], current_box)` | 与当前整图特征对齐，得到 **box_feat** | 与 `box_feat.shape[0]` 个 token 对应（常为 **1 段向量** 的长度维） |
| 拼到 `inputs_embeds` | `next_inputs_embeds = cat(embed(当前 token), box_feat)` | `query_len = box_feat.shape[0]` 个 token 追加到序列 |

`sub_image_bind=True` 时，子图经 `encode_img(sub_image)` 得到 **box_feat**，维度含义类似，但 **输入来自 crop 后的子图**，不是整图 `box_align`。

## 3. 坐标与 README 一致

- 文本里的 **`<coor> x_min,y_min,x_max,y_max</coor>`** 为 **0–1 归一化**，相对 **expand-to-square 后** 的图像（与训练一致）。
- 可视化到「原图」时，脚本用 **扩方粘贴的逆变换** 把框映射回原始宽高（见 `utils/vocot_output_viz.square_pixels_to_original`）。

## 4. 参考代码位置

- `prepare_inputs_for_generation`：`eoc_token` → `generate_box` / `generate_sub_image`
- `VolCanoMistralForCausalLM.condition_completion`：`sub_image_bind` 时设置 `cache_raw_image`

**更完整的算法梳理（多 patch、接在 `</coor>` 后的维度）：** 见 [REFBIND_ALGORITHM.md](./REFBIND_ALGORITHM.md)。
