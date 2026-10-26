"""Gallant CE 的有效样本处理；无辅助掩码时完全沿用旧 Gallant PPO。"""

import torch


class GallantCEPreparationMixin:
    """只对 CE 观测启用的采样、GAE 和损失路径。"""

    def _ce_mask(self, obs):
        mask = obs["gallant_policy_active"]
        if mask.dtype != torch.bool or mask.shape != (obs.batch_size[0], 1):
            raise ValueError("gallant_policy_active 必须为 (num_envs, 1) 布尔张量")
        return mask.squeeze(-1)

    def act(self, obs, rewards):
        if not self._ce_mode:
            return super().act(obs, rewards)
        mask = self._ce_mask(obs)
        count = obs.batch_size[0]
        width = self.storage.actions.shape[-1]
        self.transition.observations = obs
        self.transition.actions = torch.zeros(count, width, device=self.device)
        self.transition.values = torch.zeros(count, 1, device=self.device)
        self.transition.actions_log_prob = torch.zeros(count, device=self.device)
        self.transition.action_mean = torch.zeros(count, width, device=self.device)
        self.transition.action_sigma = torch.ones(count, width, device=self.device)
        extra = self.policy.extra_info
        ids = mask.nonzero(as_tuple=True)[0]
        if ids.numel():
            selected = obs[ids]
            actions, extra = self.policy.act(selected, rewards=rewards[ids])
            self.transition.actions[ids] = actions.detach()
            self.transition.values[ids] = self.policy.evaluate(selected).detach()
            self.transition.actions_log_prob[ids] = self.policy.get_actions_log_prob(actions).detach()
            self.transition.action_mean[ids] = self.policy.action_mean.detach()
            self.transition.action_sigma[ids] = self.policy.action_std.detach()
        elif self.policy.distribution is None:
            # 全准备窗口也能记录动作标准差；不执行网络或更新归一化。
            self.policy.distribution = torch.distributions.Normal(
                self.transition.action_mean, self.policy.std.clamp_min(1e-6).expand_as(self.transition.actions))
        return self.transition.actions, extra

    def compute_returns(self, obs):
        if not self._ce_mode:
            return super().compute_returns(obs)
        active = self._ce_mask(obs)
        values = torch.zeros(obs.batch_size[0], 1, device=self.device)
        ids = active.nonzero(as_tuple=True)[0]
        if ids.numel():
            values[ids] = self.policy.evaluate(obs[ids]).detach()
        # CE 的优势在 minibatch 内使用全局有效样本统计归一化。
        self.storage.compute_ce_returns(values, active[:, None], self.gamma, self.lam, False)

    def reset_ce_norm_reference(self):
        if not self._ce_mode:
            return
        norm = self.policy.vecnorm
        if norm.decay != 1.0:
            raise ValueError("CE 的多进程归一化要求 VecNorm decay=1")
        self._ce_norm_reference = (
            norm.sum.detach().clone(), norm.ssq.detach().clone(), norm.count.detach().clone())

    @torch.no_grad()
    def _ce_sync_norm(self):
        """VecNorm 的 sum/ssq 实际保存均值；归约新增的充分统计，不能平均各卡均值。"""
        if not self.is_multi_gpu:
            self.reset_ce_norm_reference()
            return
        norm = self.policy.vecnorm
        mean, second, count = self._ce_norm_reference
        delta = torch.cat(((norm.sum * norm.count - mean * count).reshape(-1),
                           (norm.ssq * norm.count - second * count).reshape(-1),
                           (norm.count - count).reshape(1)))
        torch.distributed.all_reduce(delta)
        width = norm.sum.numel()
        total = count + delta[-1]
        if total > 0:
            norm.sum.copy_((mean * count + delta[:width].reshape_as(mean)) / total)
            norm.ssq.copy_((second * count + delta[width:2*width].reshape_as(second)) / total)
        norm.count.copy_(total)
        self.reset_ce_norm_reference()

    def _ce_reduce(self, values):
        if self.is_multi_gpu:
            torch.distributed.all_reduce(values)
        return values

    def _ce_normalize_advantages(self, advantages):
        values = advantages.to(torch.float64)
        moments = torch.stack((values.new_tensor(values.numel()), values.sum(), values.square().sum()))
        self._ce_reduce(moments)
        count = moments[0]
        if count > 0 and advantages.numel():
            mean = moments[1] / count
            variance = (moments[2] / count - mean.square()).clamp_min(0)
            advantages = (advantages - mean.to(advantages.dtype)) / (
                variance.sqrt().to(advantages.dtype) + 1e-8)
        return advantages

    # 1024 环境 × 48 步 / 4 个 minibatch；更大批次仍按此上限累积梯度。
    # 演员先完成反传并释放激活，评论家再计算，避免同时保存两份大型浮点体素。
    ce_update_chunk_size = 12288

    def _update_ce_policy(self):
        self._ce_sync_norm()
        if not self.normalize_advantage_per_mini_batch:
            valid = self.storage.observations["gallant_policy_active"].bool()
            advantages = self.storage.advantages.clone()
            advantages[valid] = self._ce_normalize_advantages(advantages[valid])
            self.storage.advantages = advantages
        # 先洗牌索引，再取有效样本，避免每个 minibatch 额外复制完整体素。
        obs, actions, advantages, returns, old_log_prob, old_mu, old_sigma = [
            field.flatten(0, 1) for field in (
                self.storage.observations, self.storage.actions, self.storage.advantages,
                self.storage.returns, self.storage.actions_log_prob, self.storage.mu, self.storage.sigma)]
        valid = self._ce_mask(obs)
        totals = dict(value_function=0.0, surrogate=0.0, entropy=0.0, symmetry=0.0)
        updates = 0
        world = self.gpu_world_size if self.is_multi_gpu else 1
        for batch_ids in self.storage.gallant_batch_indices(self.num_mini_batches, self.num_learning_epochs):
            indices = batch_ids[valid[batch_ids].nonzero(as_tuple=True)[0]]
            original_size = indices.numel()
            count = self._ce_reduce(torch.tensor(float(original_size), device=self.device))
            if not count:
                continue
            selected_advantages = advantages[indices]
            if self.normalize_advantage_per_mini_batch:
                selected_advantages = self._ce_normalize_advantages(selected_advantages)
            augment = bool(self.symmetry and self.symmetry["use_data_augmentation"])
            copies = 2 if augment else 1
            width = min(self.ce_update_chunk_size, batch_ids.numel())
            kl_sum = torch.zeros((), device=self.device)
            components = torch.zeros(4, device=self.device)
            scale = world / (count * copies)
            self.optimizer.zero_grad()
            for start in range(0, original_size, width):
                ids = indices[start:start + width]
                size = ids.numel()
                # 固定前向形状以复用显存；补位只重复有效观测，所有损失和统计均排除补位。
                padded_ids = torch.cat((ids, ids[:1].expand(width - size))) if size < width else ids
                chunk_obs, chunk_actions = obs[padded_ids], actions[padded_ids]
                if augment:
                    chunk_obs, chunk_actions = self.symmetry["data_augmentation_func"](
                        obs=chunk_obs, actions=chunk_actions, env=self.symmetry["_env"])
                    if chunk_obs.batch_size[0] != width * copies:
                        raise ValueError("CE 对称扩增必须保持原 Gallant 的两份样本")
                chunk_advantages = selected_advantages[start:start + size].repeat(copies, 1).squeeze(-1)
                chunk_returns = returns[ids].repeat(copies, 1)
                chunk_log_prob = old_log_prob[ids].repeat(copies, 1).squeeze(-1)
                self.policy.act(chunk_obs)
                log_prob = self.policy.get_actions_log_prob(chunk_actions).reshape(copies, width)[:, :size].reshape(-1)
                mean, sigma = self.policy.action_mean, self.policy.action_std
                entropy_sum = self.policy.entropy.reshape(copies, width)[:, :size].sum()
                ratio = (log_prob - chunk_log_prob).exp()
                surrogate_sum = -torch.minimum(
                    chunk_advantages * ratio,
                    chunk_advantages * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)).sum()
                with torch.no_grad():
                    kl_sum += (sigma[:size].log() - old_sigma[ids].log()
                               + (old_sigma[ids].square() + (old_mu[ids] - mean[:size]).square())
                               / (2.0 * sigma[:size].square()) - 0.5).sum()
                    symmetry_sum = torch.zeros((), device=self.device)
                    if copies == 2:
                        _, mirrored = self.symmetry["data_augmentation_func"](
                            obs=None, actions=mean[:size], env=self.symmetry["_env"])
                        symmetry_sum = (mean[width:width + size] - mirrored[size:]).square().mean(-1).sum() * copies
                    components[1:] += torch.stack((surrogate_sum, entropy_sum, symmetry_sum))
                ((surrogate_sum - self.entropy_coef * entropy_sum) * scale).backward()
                # 此处演员的体素激活已释放；两组参数仍在同一次 optimizer.step 中更新。
                values = self.policy.evaluate(chunk_obs).reshape(copies, width, 1)[:, :size].reshape(-1, 1)
                value_sum = (values - chunk_returns).square().sum()
                (self.value_loss_coef * value_sum * scale).backward()
                components[0] += value_sum.detach()
            if not original_size:
                # 空卡也参加同一次梯度归约，不能因分块数不同而增加通信次数。
                loss = sum(p.sum() * 0.0 for p in self.policy.parameters() if p.requires_grad)
                loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            for group in self.optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], self.max_grad_norm)
            self.optimizer.step()
            kl = self._ce_reduce(kl_sum) / count
            if self.desired_kl is not None and self.schedule == "adaptive":
                self._apply_kl_adaptive_lr(kl)
            components = self._ce_reduce(components) / (count * copies)
            for name, value in zip(totals, components):
                totals[name] += value.item()
            updates += 1
        self.storage.clear()
        synchronize = getattr(self, "ce_statistics_sync", None)
        if synchronize is not None:
            synchronize()
        return {name: value / max(updates, 1) for name, value in totals.items()}
