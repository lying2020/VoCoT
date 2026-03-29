"""
GRPO 完整训练脚本
支持 GSM8K 数学推理任务，使用自定义奖励函数
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import argparse
import os
from reward_functions import combined_reward, DEFAULT_REWARD_FUNCTION
from data_prepare import preprocess_gsm8k

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./grpo_output")
    parser.add_argument("--use_lora", action="store_true", help="使用 LoRA 减少显存")
    parser.add_argument("--test_run", action="store_true", help="使用 100 条数据快速测试")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载数据集
    print("📊 加载数据集...")
    if args.test_run:
        train_dataset = preprocess_gsm8k("train", num_samples=100)
    else:
        train_dataset = preprocess_gsm8k("train")
    print(f"训练集大小: {len(train_dataset)}")
    
    # 2. 加载 tokenizer 和模型
    print("🤖 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 可选：使用 LoRA
    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # 3. GRPO 配置
    config = GRPOConfig(
        num_generations=args.num_generations,
        max_completion_length=args.max_length,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        beta=args.beta,
        epsilon=0.2,
        logging_steps=10,
        save_steps=500,
        output_dir=args.output_dir,
        report_to="wandb" if not args.test_run else "none",
    )
    
    # 4. 创建训练器
    print("🏋️ 初始化 GRPO Trainer...")
    trainer = GRPOTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        reward_function=combined_reward,  # 使用组合奖励函数
        tokenizer=tokenizer,
    )
    
    # 5. 开始训练
    print("🚀 开始训练...")
    trainer.train()
    
    # 6. 保存模型
    final_dir = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"✅ 模型已保存至: {final_dir}")

if __name__ == "__main__":
    main()