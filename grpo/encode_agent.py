import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 4.2 排序采样（自回归版，可获得 log_prob）
def sample_permutation(scores):
    """
    scores: (n,) 未归一化分数
    returns:
        perm: (n,) 排列索引（从大到小）
        log_prob: 标量，该排列的 log 概率
    """
    n = len(scores)
    remaining = torch.arange(n)
    perm = []
    log_prob = 0.0
    scores_copy = scores.clone()
    for i in range(n):
        # 当前剩余 token 的 logits
        logits = scores_copy[remaining]
        probs = F.softmax(logits, dim=0)
        # 采样
        dist = torch.distributions.Categorical(probs)
        idx = dist.sample()
        chosen = remaining[idx]
        perm.append(chosen.item())
        log_prob += dist.log_prob(idx)
        # 从 remaining 中移除该 token
        remaining = torch.cat([remaining[:idx], remaining[idx+1:]])
        # 不需要更新 scores_copy，因为分数不变，只是概率重新归一化
    return torch.tensor(perm), log_prob


# 4.3 奖励计算函数
def compute_reward(perm, importance, optimal_reward):
    """
    perm: (n,) 排列（从最重要到最不重要）
    importance: (n,) 每个 token 的重要性（真值）
    optimal_reward: 标量，最优排列下的总 reward（负值）
    """
    imp_sorted = importance[perm]  # 按排列顺序
    n = len(imp_sorted)
    # 计算后缀和
    suffix_sum = torch.cumsum(imp_sorted.flip(0), 0).flip(0)  # (n,)
    # 对于 b=1..n-1，淘汰的是索引 b..n-1，重要性之和 = suffix_sum[b]
    evicted_sum = suffix_sum[1:].sum()  # 对所有 b 求和
    reward = -evicted_sum
    # 归一化
    normalized_reward = reward / optimal_reward
    return normalized_reward



# 4.4 训练循环（RLOO）
def train_step(agent, optimizer, batch_data, K=8):
    """
    batch_data: 包含 keys, values, positions, hop_steps, modal_types, importance, optimal_reward
    """
    n = batch_data['keys'].shape[0]
    scores = agent(batch_data['keys'], batch_data['values'],
                   batch_data['positions'], batch_data['hop_steps'], batch_data['modal_types'])

    rewards = []
    log_probs = []
    for _ in range(K):
        perm, log_prob = sample_permutation(scores.detach())  # 注意 detach 或者使用同一个 scores？
        # 实际上 scores 应该参与梯度，但采样过程需要停止梯度？REINFORCE 中 scores 是参数化的，
        # 采样时要用当前 scores 得到分布，但梯度是通过 log_prob * advantage 回传。因此 scores 不应 detach。
        # 正确写法：perm, log_prob = sample_permutation(scores)  # scores 带梯度
        # 但 sample_permutation 内部使用了 softmax 和 categorical，会自动连接梯度。
        # 然而，由于采样是离散的，log_prob 的梯度可以通过 score function 估计。PyTorch 的 Categorical 会提供正确的梯度。
        # 所以我们直接使用 scores（不 detach）。
        perm, log_prob = sample_permutation(scores)
        reward = compute_reward(perm, batch_data['importance'], batch_data['optimal_reward'])
        rewards.append(reward)
        log_probs.append(log_prob)

    rewards = torch.stack(rewards)
    log_probs = torch.stack(log_probs)
    baseline = rewards.mean()
    advantages = rewards - baseline
    loss = -(advantages * log_probs).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

# 4.5 数据加载与预计算 optimal_reward
# 你需要预先为每个训练序列计算 optimal_reward（使用 importance 降序排列）。注意：optimal_reward 是一个负值（因为 reward 是负的 evicted importance），归一化时除以它会使 reward 在 [0,1] 之间（因为最优 reward 绝对值最大，最差的 reward 接近 0）。

# 5. 与 GRPO 的关系
# 你提到后续想使用 GRPO 进行强化学习。GRPO 是一种组相对策略优化，常用于 LLM 微调（如数学推理）。而 KVP 是专门用于 KV cache 淘汰的 RL 排序学习。两者可以结合：

# KVP 作为前置排序器：训练好的 KVP agent 可以为每个 token 输出一个“重要性分数”，这个分数可以作为 GRPO 中的 奖励信号 或 优势估计 的一部分。例如，在 GRPO 微调 MLLM 时，你可以用 KVP 分数来加权 token 的损失，或者用 KVP 淘汰不重要的 token 来节省显存，从而允许更大的 batch size 或更长的上下文。

# GRPO 优化 KVP 的策略：也可以反过来，将 KVP agent 的参数作为 GRPO 优化的目标，用 GRPO 代替 REINFORCE。但论文已证明 RL 是有效的，你可以直接沿用其方法。

# 如果你想使用 GRPO 框架来训练这个排序 agent，你需要将问题重新表述为：在每个解码步骤，agent 需要输出一个分数（或保留/淘汰动作），然后定义组内优势。但 KVP 的排序问题是全局的，GRPO 更适合 token‑level 的生成策略。因此我建议你先实现 KVP 的 REINFORCE 训练，成功后再考虑是否迁移到 GRPO。

# 6. 总结与建议步骤
# 数据采集：运行你的 MLLM 在多跳推理任务上，保存每个推理样本的：

# 所有输入 token 的 (k, v, position, hop_step, modal_type)

# 后续生成过程中每个历史 token 的未来注意力总和（或自定义重要性）

# 预处理：按 head 分组，为每个 head 构建独立的训练数据集。计算每个序列的 optimal_reward。

# 实现 KVP agent：每个 head 一个 2‑层 MLP，输入为 (k, v, position, hop_step, modal_type) 的拼接。

# 离线训练：使用 REINFORCE + RLOO，每个 step 采样 K 个排列，计算归一化 reward，更新 agent。

# 评估：在保留的测试集上，测量 agent 排序的 -R^b（被淘汰 token 的重要性之和）与最优排序的差距。

# 部署：推理时，将 agent 的 MLP 加载到每个 head，对当前 KV cache 的所有 token 计算分数，用 argsort 得到排序，按预算保留 top‑b 个 token。注意：论文中始终保留前 4 个和最后 16 个 token（可以借鉴），防止关键 token 被误淘汰。

# 注意事项：

# 训练时不需要 LLM 在线参与，完全离线，效率高。

# 每个 head 独立训练，参数很少（~600K），可以并行。

# 如果你的 MLLM 使用了 GQA（Grouped Query Attention），需要按 group 训练 agent，而不是每个 query head（论文中按 KV head 训练，数量较少）。

# 如果 future importance 计算成本过高，可以先在小规模数据集上实验，验证效果。

# 希望这个梳理对你有帮助。如果需要更具体的代码（如完整的数据预处理脚本、多 head 并行训练），我可以进一步提供。



# 4.1 特征编码器与 Agent
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x, pos):
        # x: (batch, n, d_model) 用于广播
        return x + self.pe[pos]   # pos: (n,)

class KVAgent(nn.Module):
    def __init__(self, k_dim, v_dim, pos_dim=32, hop_embed_dim=16, modal_embed_dim=8, hidden_dim=256):
        super().__init__()
        self.pos_enc = PositionalEncoding(pos_dim)
        self.hop_embed = nn.Embedding(num_embeddings=10, embedding_dim=hop_embed_dim)  # 假设最多10跳
        self.modal_embed = nn.Embedding(num_embeddings=2, embedding_dim=modal_embed_dim)

        input_dim = k_dim + v_dim + pos_dim + hop_embed_dim + modal_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出标量分数
        )

    def forward(self, keys, values, positions, hop_steps, modal_types):
        """
        keys: (n, k_dim)
        values: (n, v_dim)
        positions: (n,) 整数
        hop_steps: (n,) 整数
        modal_types: (n,) 整数 (0:text, 1:visual)
        """
        # 位置编码（此处直接加正弦编码，也可学习）
        pos_enc = self.pos_enc.pe[positions]  # (n, pos_dim)
        hop_emb = self.hop_embed(hop_steps)   # (n, hop_dim)
        modal_emb = self.modal_embed(modal_types)  # (n, modal_dim)
        features = torch.cat([keys, values, pos_enc, hop_emb, modal_emb], dim=-1)
        scores = self.mlp(features).squeeze(-1)  # (n,)
        return scores