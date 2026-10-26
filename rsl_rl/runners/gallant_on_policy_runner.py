"""Gallant 算法构造与训练入口；环境统计由下游适配层提供。"""

from rsl_rl.algorithms import GallantPPO
from rsl_rl.modules import GallantEncoderConcat, resolve_symmetry_config

from .on_policy_runner import OnPolicyRunner


class GallantOnPolicyRunner(OnPolicyRunner):
    def _construct_algorithm(self, obs):
        if self.cfg.get("amp_cfg") or self.alg_cfg.get("rnd_cfg"):
            raise ValueError("Gallant runner does not support AMP or RND.")
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        policy_cfg = dict(self.policy_cfg)
        algorithm_cfg = dict(self.alg_cfg)
        if policy_cfg.pop("class_name") != "GallantEncoderConcat":
            raise ValueError("Gallant runner requires GallantEncoderConcat.")
        if algorithm_cfg.pop("class_name") != "GallantPPO":
            raise ValueError("Gallant runner requires GallantPPO.")
        policy = GallantEncoderConcat(
            obs, self.cfg["obs_groups"], self.env.num_actions,
            env_cfg=self.env.cfg, alg_cfg=self.alg_cfg, **policy_cfg,
        ).to(self.device)
        algorithm = GallantPPO(
            policy, device=self.device, train_cfg=self.cfg,
            multi_gpu_cfg=self.multi_gpu_cfg, **algorithm_cfg,
        )
        algorithm.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return algorithm

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # 普通 Gallant 训练从同步 reset 开始，不随机打散首次回合长度。
        return super().learn(num_learning_iterations, init_at_random_ep_len=False)
