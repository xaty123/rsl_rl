"""Gallant rollout 的初始状态掩码与逐 epoch 洗牌。"""

import torch

from .rollout_storage import RolloutStorage


class GallantRolloutStorage(RolloutStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_init = torch.zeros_like(self.dones, dtype=torch.bool)

    def gallant_batches(self, num_mini_batches, num_epochs):
        size = self.num_envs * self.num_transitions_per_env
        if size % num_mini_batches:
            raise ValueError("Gallant rollout size must be divisible by num_mini_batches.")
        width = size // num_mini_batches
        fields = (
            self.observations, self.actions, self.advantages, self.returns,
            self.actions_log_prob, self.mu, self.sigma, self.is_init,
        )
        fields = [field.flatten(0, 1) for field in fields]
        for _ in range(num_epochs):
            # 原版 make_batch 在每个 epoch 内重新洗牌。
            indices = torch.randperm(size, device=self.device)
            for start in range(0, size, width):
                batch = indices[start:start + width]
                yield tuple(field[batch] for field in fields)

