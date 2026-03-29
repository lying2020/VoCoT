"""
数据预处理脚本：将 GSM8K 数据集转换为 GRPO 所需格式
"""
from datasets import load_dataset
from typing import Dict, List

def preprocess_gsm8k(split: str = "train", num_samples: int = None):
    """
    加载并预处理 GSM8K 数据集
    Args:
        split: "train" 或 "test"
        num_samples: 限制样本数量（用于快速测试）
    Returns:
        HuggingFace Dataset 对象
    """
    dataset = load_dataset("openai/gsm8k", "main")
    raw = dataset[split]
    
    def format_example(example: Dict) -> Dict:
        # 答案在 "####" 之后
        ground_truth = example['answer'].split("####")[-1].strip()
        return {
            "prompt": example['question'],
            "ground_truth": ground_truth
        }
    
    processed = raw.map(format_example)
    if num_samples:
        processed = processed.select(range(min(num_samples, len(processed))))
    return processed

def create_custom_dataset(prompts: List[str], answers: List[str]):
    """
    创建自定义数据集（用于非标准任务）
    Args:
        prompts: 问题列表
        answers: 标准答案列表
    Returns:
        Dataset 对象
    """
    from datasets import Dataset
    data = {"prompt": prompts, "ground_truth": answers}
    return Dataset.from_dict(data)

if __name__ == "__main__":
    # 示例：预处理训练集前100条
    train_data = preprocess_gsm8k("train", num_samples=100)
    print(train_data[0])
    # 保存为 JSON 文件
    train_data.to_json("train_data.json")