# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

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
