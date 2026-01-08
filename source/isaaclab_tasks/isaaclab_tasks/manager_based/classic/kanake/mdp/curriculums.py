# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)




def modify_command_resampling_time(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    command_name: str, 
    resampling_time_range: tuple[float, float], 
    num_steps: int
):
    """Curriculum that modifies command resampling time range after a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        command_name: The name of the command term.
        resampling_time_range: The new resampling time range (min_time, max_time).
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain command term
        command_term = env.command_manager.get_term(command_name)
        # update resampling time range
        command_term.resampling_time_range = resampling_time_range
        # print(f"[CURRICULUM] Changed '{command_name}' resampling time to {resampling_time_range} at step {env.common_step_counter}")


def modify_command_resampling_time_gradual(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    command_name: str, 
    initial_range: tuple[float, float],
    target_range: tuple[float, float], 
    start_steps: int,
    end_steps: int
):
    """Curriculum that gradually changes command resampling time range over a period.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        command_name: The name of the command term.
        initial_range: The initial resampling time range (min_time, max_time).
        target_range: The target resampling time range (min_time, max_time).
        start_steps: The step count when gradual change starts.
        end_steps: The step count when gradual change ends.
    """
    current_step = env.common_step_counter
    
    if current_step >= start_steps:
        if current_step >= end_steps:
            # Use target range
            new_range = target_range
        else:
            # Linear interpolation between initial and target
            progress = (current_step - start_steps) / (end_steps - start_steps)
            min_time = initial_range[0] + progress * (target_range[0] - initial_range[0])
            max_time = initial_range[1] + progress * (target_range[1] - initial_range[1])
            new_range = (min_time, max_time)
        
        # Update command term
        command_term = env.command_manager.get_term(command_name)
        command_term.cfg.resampling_time_range = new_range
        
        # Log every 1000 steps
        if current_step % 1000 == 0:
            print(f"[CURRICULUM] '{command_name}' resampling time: {new_range} at step {current_step}")


def modify_command_relative_envs(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    command_name: str, 
    rel_standing_envs: float,
    rel_heading_envs: float,
    num_steps: int
):
    """Curriculum that modifies the relative number of standing/heading environments.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        command_name: The name of the command term.
        rel_standing_envs: New ratio of standing environments.
        rel_heading_envs: New ratio of heading environments.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain command term
        command_term = env.command_manager.get_term(command_name)
        # update relative environment ratios
        if hasattr(command_term.cfg, 'rel_standing_envs'):
            command_term.cfg.rel_standing_envs = rel_standing_envs
        if hasattr(command_term.cfg, 'rel_heading_envs'):
            command_term.cfg.rel_heading_envs = rel_heading_envs
        
        print(f"[CURRICULUM] Changed '{command_name}' env ratios - standing: {rel_standing_envs}, heading: {rel_heading_envs}")



def terrain_levels_pose(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    target_command_name: str = "kanake_command",  
    success_tolerance: float = 0.2,          # 목표 반경 0.2m 안에 들어오면 성공
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Curriculum based on distance to the target pose.
    목표 지점에 도착하면(거리 < tolerance) 지형 난이도를 높이고, 실패하면 낮춥니다.
    """
    # 1. 로봇과 지형 객체 가져오기
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    
    # 2. 커맨드 객체(Term) 자체를 가져옵니다. (커맨드 텐서가 아니라 객체를 가져와야 world 타겟을 알 수 있음)
    # 님이 만든 KanakeBaseCommand 클래스의 인스턴스를 가져오는 것입니다.
    command_term = env.command_manager.get_term(target_command_name)
    
    # 3. 목표 위치(World Frame)와 현재 로봇 위치 가져오기
    # KanakeBaseCommand에 self.pos_command_w가 정의되어 있어야 합니다.
    target_pos_w = command_term.pos_command_w[env_ids, :2] 
    current_pos_w = asset.data.root_pos_w[env_ids, :2]

    # 4. 목표까지의 남은 거리 계산
    distance_to_target = torch.norm(target_pos_w - current_pos_w, dim=1)

    # 5. 레벨 조정 로직
    # 성공: 목표 반경 안에 들어왔는가? -> 난이도 UP
    move_up = distance_to_target < success_tolerance
    
    # 실패: 성공하지 못했으면 -> 난이도 DOWN
    move_down = ~move_up 

    # 6. 지형 레벨 업데이트
    terrain.update_env_origins(env_ids, move_up, move_down)

    return torch.mean(terrain.terrain_levels.float())