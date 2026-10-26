"""复现当前 Gallant PPO 的损失、更新顺序和归一化约定。"""

import torch
from tensordict import TensorDict

from rsl_rl.storage import GallantRolloutStorage

from .ppo import PPO


class GallantPPO(PPO):
    """仅用于非循环、双编码器的 ppo_gallant_concat。"""

    def __init__(self, policy, adapt_lr_actor_only: bool = False, **kwargs):
        super().__init__(policy, **kwargs)
        self.adapt_lr_actor_only = adapt_lr_actor_only
        if self.use_clipped_value_loss or self.rnd or self.policy.is_recurrent:
            raise ValueError("GallantPPO requires ordinary MSE, no RND, and a feedforward policy.")
        if self.symmetry and self.symmetry["use_mirror_loss"]:
            raise ValueError("GallantPPO uses symmetric samples, without a mirror loss.")

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        self.storage = GallantRolloutStorage(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, self.device
        )
        self._is_init = torch.ones(num_envs, 1, dtype=torch.bool, device=self.device)

    def process_env_step(self, obs, rewards, dones, extras):
        self.storage.is_init[self.storage.step].copy_(self._is_init)
        # timeouts 应对终止时的物理状态 bootstrap，而不是 reset 后的零观测。
        extras = dict(extras)
        time_outs = extras.pop("time_outs", None)
        if self.clip_negative_rewards:
            rewards = rewards.clamp_min(0.0)
        if time_outs is not None and time_outs.any():
            terminal = extras.get("gallant_terminal_observations")
            ids = extras.get("gallant_terminal_env_ids")
            if terminal is None or ids is None:
                raise RuntimeError("Gallant timeout is missing terminal observations.")
            ids = ids.to(self.device)
            terminal = TensorDict(terminal, batch_size=[len(ids)]).to(self.device)
            with torch.no_grad():
                values = self.policy.evaluate(terminal).squeeze(-1)
            rewards = rewards.clone()
            rewards[ids] += self.gamma * values * time_outs.to(self.device)[ids]
        # 已按原版顺序先截奖励，再添加 bootstrap，避免父类再次截断价值。
        clip_rewards = self.clip_negative_rewards
        self.clip_negative_rewards = False
        try:
            super().process_env_step(obs, rewards, dones, extras)
        finally:
            self.clip_negative_rewards = clip_rewards
        self._is_init.copy_(dones.reshape(-1, 1).bool())

    def _apply_kl_adaptive_lr(self, kl_mean):
        if self.is_multi_gpu:
            # 各卡 minibatch 等量；保留调用方的本地 KL，仅归约副本。
            kl_mean = kl_mean.clone()
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
        if not self.is_multi_gpu or self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
                self.learning_rate = min(self.adaptive_lr_ceil, self.learning_rate * self.adaptive_lr_grow)
        if self.is_multi_gpu:
            lr = torch.tensor(self.learning_rate, dtype=torch.float64, device=self.device)
            torch.distributed.broadcast(lr, src=0)
            self.learning_rate = lr.item()
        groups = self.optimizer.param_groups[:1] if self.adapt_lr_actor_only else self.optimizer.param_groups
        for group in groups:
            group["lr"] = self.learning_rate

    def _update_policy(self):
        totals = dict(value_function=0.0, surrogate=0.0, entropy=0.0, symmetry=0.0)
        updates = 0
        for obs, actions, advantages, returns, old_log_prob, old_mu, old_sigma, is_init in self.storage.gallant_batches(
            self.num_mini_batches, self.num_learning_epochs
        ):
            original_size = obs.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                obs, actions = self.symmetry["data_augmentation_func"](
                    obs=obs, actions=actions, env=self.symmetry["_env"]
                )
            copies = obs.batch_size[0] // original_size
            valid = (~is_init).repeat(copies, 1).squeeze(-1)
            advantages = advantages.repeat(copies, 1).squeeze(-1)
            returns = returns.repeat(copies, 1)
            old_log_prob = old_log_prob.repeat(copies, 1).squeeze(-1)
            self.policy.act(obs)
            log_prob = self.policy.get_actions_log_prob(actions)
            values = self.policy.evaluate(obs)
            mean = self.policy.action_mean
            sigma = self.policy.action_std
            entropy = self.policy.entropy.mean()
            ratio = (log_prob - old_log_prob).exp()
            surrogate = -torch.minimum(
                advantages * ratio,
                advantages * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param),
            )
            # 掩码后对整批取 mean，不能除以有效样本数；熵不做 is_init 掩码。
            surrogate_loss = (surrogate * valid).mean()
            value_loss = ((values - returns).square() * valid.unsqueeze(-1)).mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            with torch.no_grad():
                kl = (
                    sigma[:original_size].log() - old_sigma.log()
                    + (old_sigma.square() + (old_mu - mean[:original_size]).square())
                    / (2.0 * sigma[:original_size].square()) - 0.5
                ).sum(-1).mean()
                symmetry_loss = torch.zeros((), device=self.device)
                if copies == 2:
                    _, mirrored = self.symmetry["data_augmentation_func"](
                        obs=None, actions=mean[:original_size], env=self.symmetry["_env"]
                    )
                    symmetry_loss = (mean[original_size:] - mirrored[original_size:]).square().mean()
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            # AdamW 的两个参数组与原版 actor/critic 的参数边界一致。
            for group in self.optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], self.max_grad_norm)
            self.optimizer.step()
            if self.desired_kl is not None and self.schedule == "adaptive":
                self._apply_kl_adaptive_lr(kl)
            for name, value in zip(totals, (value_loss, surrogate_loss, entropy, symmetry_loss)):
                totals[name] += value.item()
            updates += 1
        if self.is_multi_gpu:
            for name in ("sum", "ssq", "count"):
                torch.distributed.broadcast(getattr(self.policy.vecnorm, name), src=0)
        self.storage.clear()
        return {name: value / updates for name, value in totals.items()}
