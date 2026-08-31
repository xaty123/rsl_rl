# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import torch
import torch.nn as nn
from torch import autograd
from rsl_rl.utils import Normalizer

class AMPDiscriminator(nn.Module):
    """
    判别器神经网络用于对抗式运动先验（AMP）奖励预测。
    
    参数:
        input_dim (int): 输入特征向量的维度（拼接的状态和下一个状态）。
        amp_reward_coef (float): 用于缩放AMP奖励的系数。
        hidden_layer_sizes (list[int]): MLP主干中隐藏层的大小列表。
        device (torch.device): 用于运行模型的设备（CPU或GPU）。
        task_reward_lerp (float, 可选): AMP奖励与任务奖励之间的插值因子。
            默认为0.0（只使用AMP奖励）。
    属性:
        trunk (nn.Sequential): 处理输入特征的MLP层。
        amp_linear (nn.Linear): 产生判别器输出的最终线性层。
        task_reward_lerp (float): 用于组合奖励的插值因子。
    """

    def __init__(self, input_dim, amp_reward_coef, hidden_layer_sizes, device, normalize, task_reward_lerp=0.0):
        super().__init__()

        self.device = device
        self.input_dim = input_dim

        self.amp_reward_coef = amp_reward_coef
        amp_layers = []
        curr_in_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            amp_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            amp_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.trunk = nn.Sequential(*amp_layers).to(device)
        self.amp_linear = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.amp_linear.train()

        self.task_reward_lerp = task_reward_lerp
        
        self.normalizer = Normalizer(int(input_dim/2)) if normalize else None

    def forward(self, x):
        """
        Forward pass through the discriminator network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Discriminator output logits with shape (batch_size, 1).
        """
        h = self.trunk(x)
        d = self.amp_linear(h)
        return d

    def compute_grad_pen(self, expert_state, expert_next_state, lambda_=10):
        """
        Compute gradient penalty for the expert data, used to regularize the discriminator.

        Args:
            expert_state (torch.Tensor): Batch of expert states.
            expert_next_state (torch.Tensor): Batch of expert next states.
            lambda_ (float, optional): Gradient penalty coefficient. Defaults to 10.

        Returns:
            torch.Tensor: Scalar gradient penalty loss.
        """
        expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
        expert_data.requires_grad = True

        disc = self.amp_linear(self.trunk(expert_data))
        ones = torch.ones(disc.size(), device=disc.device)
        grad = autograd.grad(
            outputs=disc, inputs=expert_data, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        # Enforce that the grad norm approaches 0.
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def predict_amp_reward(self, state, next_state, task_reward):
        """
        Predict the AMP reward given current and next states, optionally interpolated with a task reward.

        Args:
            state (torch.Tensor): Current state tensor.
            next_state (torch.Tensor): Next state tensor.
            task_reward (torch.Tensor): Task-specific reward tensor.

        Returns:
            tuple:
                - reward (torch.Tensor): Predicted AMP reward (optionally interpolated) with shape (batch_size,).
                - d (torch.Tensor): Raw discriminator output logits with shape (batch_size, 1).
        """
        with torch.no_grad():
            self.eval()
            if self.normalizer is not None:
                state = self.normalizer.normalize_torch(state, self.device)
                next_state = self.normalizer.normalize_torch(next_state, self.device)

            d = self.amp_linear(self.trunk(torch.cat([state, next_state], dim=-1)))
            style_reward = self.amp_reward_coef * torch.clamp(1 - (1 / 4) * torch.square(d - 1), min=0)
            if self.task_reward_lerp > 0:
                final_reward, disc_r, task_r= self._lerp_reward(style_reward, task_reward.unsqueeze(-1))
            self.train()
        return final_reward.squeeze(), style_reward.squeeze(), d, disc_r.squeeze(), task_r.squeeze()

    def _lerp_reward(self, disc_r, task_r):
        """
        Linearly interpolate between discriminator reward and task reward.

        Args:
            disc_r (torch.Tensor): Discriminator reward.
            task_r (torch.Tensor): Task reward.

        Returns:
            torch.Tensor: Interpolated reward.
        """
        r = (1.0 - self.task_reward_lerp) * disc_r + self.task_reward_lerp * task_r
        disc_r = (1.0 - self.task_reward_lerp) * disc_r
        task_r = self.task_reward_lerp * task_r
        return r, disc_r, task_r
