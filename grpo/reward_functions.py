"""
GRPO 奖励函数模块
支持数学题的正确性奖励、格式奖励以及组合奖励
"""
import re
from typing import List, Union

def format_reward(completions: List[str]) -> List[float]:
    """
    格式奖励：要求答案放在 \\boxed{} 中
    """
    pattern = r'\\boxed\{.*?\}'
    rewards = []
    for completion in completions:
        if re.search(pattern, completion):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def correctness_reward(completions: List[str], ground_truths: List[str]) -> List[float]:
    """
    正确性奖励：提取 \\boxed{} 内的答案并与标准答案比较
    支持整数、浮点数和字符串比较
    """
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        match = re.search(r'\\boxed\{([^}]*)\}', completion)
        if match:
            predicted = match.group(1).strip()
            try:
                # 数值比较（处理整数、浮点数）
                if abs(float(predicted) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except ValueError:
                # 字符串比较
                rewards.append(1.0 if predicted == gt else 0.0)
        else:
            rewards.append(0.0)
    return rewards

def combined_reward(completions: List[str], ground_truths: List[str], **kwargs) -> List[float]:
    """
    组合奖励：格式正确 + 答案正确（只有两者都对才给满分）
    """
    format_scores = format_reward(completions)
    correct_scores = correctness_reward(completions, ground_truths)
    return [f * c for f, c in zip(format_scores, correct_scores)]

def soft_reward(completions: List[str], ground_truths: List[str]) -> List[float]:
    """
    软奖励：部分匹配时给予中间分数
    例如：格式正确给0.3，答案正确额外给0.7
    """
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        score = 0.0
        # 格式分
        if re.search(r'\\boxed\{.*?\}', completion):
            score += 0.3
        # 答案分
        match = re.search(r'\\boxed\{([^}]*)\}', completion)
        if match:
            predicted = match.group(1).strip()
            try:
                if abs(float(predicted) - float(gt)) < 1e-5:
                    score += 0.7
            except:
                if predicted == gt:
                    score += 0.7
        rewards.append(score)
    return rewards

# 你可以根据需要选择不同的奖励函数
DEFAULT_REWARD_FUNCTION = combined_reward