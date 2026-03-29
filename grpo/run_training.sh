#!/bin/bash
# GRPO 训练启动脚本

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用单卡，可改为 0,1,2,3 多卡
export WANDB_MODE=disabled      # 如需 wandb，改为 online

# 创建输出目录
mkdir -p ./grpo_output

# 运行训练
python grpo_train.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --num_generations 8 \
    --learning_rate 5e-6 \
    --batch_size 2 \
    --beta 0.04 \
    --epochs 1 \
    --max_length 256 \
    --output_dir ./grpo_output \
    --test_run   # 快速测试用100条数据，正式训练去掉此参数

# 如需使用 LoRA，添加 --use_lora
