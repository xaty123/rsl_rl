"""G1-Gallant 的 EncoderConcat。

主实验用二维卷积。grid_map_ 是 (N, 40, 32, 32)：40 是通道，32×32 是空间。
演员和评论家各一份编码器，权重不共享。两边读同一份归一化后的 policy，再加上体素。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal


class VecNorm(nn.Module):
    """累计 policy 的均值和方差。decay=1 时不遗忘旧样本。"""

    def __init__(self, size: int, decay: float = 1.0, eps: float = 1e-5) -> None:
        super().__init__()
        self.decay = float(decay)
        self.eps = eps
        self.register_buffer("sum", torch.zeros(size))
        self.register_buffer("ssq", torch.zeros(size))
        self.register_buffer("count", torch.tensor(0.0))

    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().reshape(-1, values.shape[-1])
        batch = flat.shape[0]
        if batch == 0:
            return
        if self.decay < 1.0:
            self.count.mul_(self.decay).add_(batch)
            self.sum.mul_(self.decay).add_(flat.sum(0))
            self.ssq.mul_(self.decay).add_(flat.square().sum(0))
            return
        self.count.add_(batch)
        weight = batch / self.count
        self.sum.lerp_(flat.mean(0), weight)
        self.ssq.lerp_(flat.square().mean(0), weight)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        if float(self.count) <= 0.0:
            return values
        if self.decay < 1.0:
            mean = self.sum / self.count
            var = (self.ssq / self.count - mean.square()).clamp_min(self.eps)
        else:
            mean = self.sum
            var = (self.ssq - mean.square()).clamp_min(self.eps)
        return (values - mean) / var.sqrt()


class ConcatEncoder(nn.Module):
    """本体编到 256，体素编到 64，拼成 320 后再融合到 256。"""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LazyLinear(256),
            nn.Mish(),
            nn.LayerNorm(256),
            nn.LazyLinear(256),
        )
        self.cnn = nn.Sequential(
            nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1),
            nn.Mish(),
            nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1),
            nn.Mish(),
            nn.LazyConv2d(8, kernel_size=3, stride=2, padding=1),
            nn.Mish(),
            nn.Flatten(),
        )
        self.cnn_head = nn.Sequential(
            nn.LazyLinear(64),
            nn.Mish(),
            nn.LayerNorm(64),
            nn.LazyLinear(64),
        )
        self.fuse = nn.Sequential(nn.Mish(), nn.LazyLinear(256), nn.Mish())

    def forward(self, policy: torch.Tensor, voxels: torch.Tensor) -> torch.Tensor:
        voxel_feature = self.cnn_head(self.cnn(voxels.float()))
        proprio_feature = self.mlp(policy)
        return self.fuse(torch.cat([proprio_feature, voxel_feature], dim=-1))


class EncoderConcat(nn.Module):
    """演员和评论家各一份 ConcatEncoder。policy 用一套共享的 VecNorm。"""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        env_cfg: Any = None,
        alg_cfg: dict | None = None,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "mish",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "EncoderConcat.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs))
            )
        if state_dependent_std or noise_std_type != "scalar":
            raise ValueError("EncoderConcat uses one learnable std vector of length num_actions.")
        super().__init__()
        self.obs_groups = obs_groups
        self.extra_info: dict = {}
        policy = obs["policy"]
        voxels = obs["grid_map_"]
        if policy.shape[-1] != 504:
            raise ValueError(f"EncoderConcat expects a 504-D policy. Received {tuple(policy.shape)}.")
        if tuple(voxels.shape[-3:]) != (40, 32, 32):
            raise ValueError(f"EncoderConcat expects voxels (40, 32, 32). Received {tuple(voxels.shape)}.")

        self.vecnorm = VecNorm(policy.shape[-1], decay=1.0)
        self.actor_encoder = ConcatEncoder()
        self.critic_encoder = ConcatEncoder()
        self.actor_mean = nn.Linear(256, num_actions)
        self.critic_head = nn.Linear(256, 1)
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        self._accumulate_norm = True
        Normal.set_default_validate_args(False)
        self.to(policy.device)

        example_policy = policy[:1].float()
        example_voxels = voxels[:1]
        self.actor_encoder(example_policy, example_voxels)
        self.critic_encoder(example_policy, example_voxels)
        self.apply(self._init_module)
        nn.init.orthogonal_(self.actor_mean.weight, 0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        print(f"EncoderConcat: policy {tuple(policy.shape)} voxels {tuple(voxels.shape)} actions {num_actions}")

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, 0.1)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, 0.1)
            nn.init.constant_(module.bias, 0.0)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def set_norm_update(self, enabled: bool) -> None:
        """采样时累计归一化。PPO 更新策略时关掉，避免同一批样本被反复计入。"""
        self._accumulate_norm = enabled

    def _policy(self, obs: TensorDict, update: bool) -> torch.Tensor:
        policy = obs["policy"]
        if update and self.training and self._accumulate_norm:
            self.vecnorm.update(policy)
        return self.vecnorm.normalize(policy)

    def _update_distribution(self, obs: TensorDict, update_norm: bool) -> None:
        feature = self.actor_encoder(self._policy(obs, update=update_norm), obs["grid_map_"])
        mean = self.actor_mean(feature)
        self.distribution = Normal(mean, self.std.expand_as(mean))

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        self._update_distribution(obs, update_norm=True)
        return self.distribution.sample(), self.extra_info

    def act_inference(self, obs: TensorDict) -> tuple[torch.Tensor, dict]:
        feature = self.actor_encoder(self._policy(obs, update=False), obs["grid_map_"])
        return self.actor_mean(feature), self.extra_info

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        feature = self.critic_encoder(self._policy(obs, update=False), obs["grid_map_"])
        return self.critic_head(feature)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        return

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True

    def create_optimizers(self, learning_rate: float) -> dict[str, torch.optim.Optimizer]:
        return {"optimizer": torch.optim.Adam(self.parameters(), lr=learning_rate)}


class GallantEncoderConcat(EncoderConcat):
    """保留网络与 checkpoint 键，集中 Gallant 的归一化、标准差、优化器和导出。"""

    def __init__(self, obs, *args, **kwargs):
        super().__init__(obs, *args, **kwargs)
        self.vecnorm.count.fill_(1.0)
        # 原版构网时对 observation_spec.zero() 做一次更新，包含整批环境。
        self.vecnorm.update(torch.zeros_like(obs["policy"]))

    def _update_distribution(self, obs, update_norm):
        feature = self.actor_encoder(self._policy(obs, update=update_norm), obs["grid_map_"])
        mean = self.actor_mean(feature)
        self.distribution = Normal(mean, self.std.clamp_min(1e-6).expand_as(mean))

    def create_optimizers(self, learning_rate: float) -> dict[str, torch.optim.Optimizer]:
        """一个 AdamW，权重衰减 0.01。第 0 组是演员，第 1 组是评论家。

        成功版本的 actor_std 属于演员。这里的 std 同样放进第 0 组，学习率和衰减都跟着演员。
        VecNorm 只有 buffer，不在这两组里。
        """
        actor_params = (
            list(self.actor_encoder.parameters())
            + list(self.actor_mean.parameters())
            + [self.std]
        )
        critic_params = list(self.critic_encoder.parameters()) + list(self.critic_head.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": actor_params},
                {"params": critic_params},
            ],
            lr=learning_rate,
            weight_decay=0.01,
        )
        return {"optimizer": optimizer}

    def export_to_onnx(
        self,
        path: str,
        filename: str = "policy.onnx",
        normalizer: nn.Module | None = None,
        verbose: bool = False,
    ) -> None:
        """将演员导出为 ONNX。本体归一化使用本网络的 VecNorm，不使用传入的 normalizer。"""
        import os

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        exporter = _OnnxEncoderConcatExporter(self, verbose)
        exporter.export(path, filename)


class _OnnxEncoderConcatExporter(nn.Module):
    """EncoderConcat 的 ONNX 导出器。输入是未归一化的 policy 和 grid_map_。"""

    def __init__(self, policy: EncoderConcat, verbose: bool = False) -> None:
        super().__init__()
        import copy

        self.verbose = verbose
        self.actor_encoder = copy.deepcopy(policy.actor_encoder)
        self.actor_mean = copy.deepcopy(policy.actor_mean)
        vec = policy.vecnorm
        mean = vec.sum.detach().float().cpu()
        var = (vec.ssq.detach().float().cpu() - mean.square()).clamp_min(vec.eps)
        self.register_buffer("obs_mean", mean)
        self.register_buffer("obs_std", var.sqrt())

    def forward(self, policy: torch.Tensor, voxels: torch.Tensor) -> torch.Tensor:
        policy = (policy - self.obs_mean) / self.obs_std
        feature = self.actor_encoder(policy, voxels)
        return self.actor_mean(feature)

    def export(self, path: str, filename: str) -> None:
        import os

        self.to("cpu")
        self.eval()
        torch.onnx.export(
            self,
            (torch.zeros(1, 504), torch.zeros(1, 40, 32, 32)),
            os.path.join(path, filename),
            export_params=True,
            opset_version=18,
            verbose=self.verbose,
            input_names=["policy", "grid_map"],
            output_names=["actions"],
            dynamic_axes={},
        )
