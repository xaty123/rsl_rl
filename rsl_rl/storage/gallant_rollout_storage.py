"""Gallant rollout 的初始状态掩码与逐 epoch 洗牌。"""

import torch

from .rollout_storage import RolloutStorage


class GallantRolloutStorage(RolloutStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_init = torch.zeros_like(self.dones, dtype=torch.bool)

    def gallant_batch_indices(self, num_mini_batches, num_epochs):
        size = self.num_envs * self.num_transitions_per_env
        if size % num_mini_batches:
            raise ValueError("Gallant rollout size must be divisible by num_mini_batches.")
        width = size // num_mini_batches
        for _ in range(num_epochs):
            # 原版 make_batch 在每个 epoch 内重新洗牌。
            indices = torch.randperm(size, device=self.device)
            for start in range(0, size, width):
                yield indices[start:start + width]

    def gallant_batches(self, num_mini_batches, num_epochs):
        fields = (
            self.observations, self.actions, self.advantages, self.returns,
            self.actions_log_prob, self.mu, self.sigma, self.is_init,
        )
        fields = [field.flatten(0, 1) for field in fields]
        for batch in self.gallant_batch_indices(num_mini_batches, num_epochs):
            yield tuple(field[batch] for field in fields)


    def compute_ce_returns(self, last_values, last_active, gamma, lam, normalize_advantage):
        """准备行不生成优势，回合结束和准备边界均阻断 GAE。"""
        valid = self.observations["gallant_policy_active"].bool()
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            next_active = last_active if step == self.num_transitions_per_env - 1 else valid[step + 1]
            continuation = (~self.dones[step].bool()) & next_active
            bootstrap = torch.where(continuation, next_values, 0.0)
            delta = self.rewards[step] + gamma * bootstrap - self.values[step]
            advantage = torch.where(
                valid[step], delta + gamma * lam * continuation * advantage, 0.0)
            self.returns[step] = torch.where(valid[step], advantage + self.values[step], 0.0)
        self.advantages = torch.where(valid, self.returns - self.values, 0.0)
        if normalize_advantage:
            values = self.advantages[valid]
            if values.numel():
                self.advantages[valid] = (values - values.mean()) / (
                    values.std(unbiased=False) + 1e-8)
