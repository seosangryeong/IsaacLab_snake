# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab.utils import configclass


@configclass
class kanakePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # num_steps_per_env = 32
    num_steps_per_env = 64
    max_iterations = 5000
    save_interval = 50
    experiment_name = "kanake"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # algorithm = RslRlPpoAlgorithmCfg(
    #     value_loss_coef=1.0,
    #     use_clipped_value_loss=True,
    #     clip_param=0.2,
    #     entropy_coef=0.0,
    #     num_learning_epochs=5,
    #     num_mini_batches=4,
    #     learning_rate=1.0e-3,
    #     schedule="adaptive",
    #     gamma=0.99,
    #     lam=0.95,
    #     desired_kl=0.01,
    #     max_grad_norm=1.0,
    # )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,  # 0.2에서 0.1로 감소
        entropy_coef=0.01,  # 0.0에서 0.01로 증가 (다양한 동작 탐색 촉진)
        num_learning_epochs=8,  # 5에서 8로 증가 (데이터 활용도 높임)
        num_mini_batches=4,
        learning_rate=1.0e-3,  # 1e-3에서 5e-4로 감소 (더 안정적인 학습)
        schedule="adaptive",
        gamma=0.99,  # 0.99에서 0.995로 증가 (장기적 보상 더 중요시)
        lam=0.95,
        desired_kl=0.008,  # 0.01에서 0.008로 감소 (더 엄격한 KL 제한)
        max_grad_norm=0.5,  # 1.0에서 0.5로 감소 (그래디언트 업데이트 안정화)
    )