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
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


class obstacle_count_curriculum(ManagerTermBase):
    """Increase the number of active obstacles when command tracking is good enough."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._active_count = int(cfg.params.get("initial_count", 0))
        self._last_increase_step = 0
        self._num_calls = 0
        setattr(env, cfg.params.get("active_count_attr", "_teraffe_active_obstacle_count"), self._active_count)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        active_count_attr: str = "_teraffe_active_obstacle_count",
        initial_count: int = 0,
        reward_terms: tuple[str, ...] = ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"),
        thresholds: tuple[float, ...] = (1.4, 1.6, 1.8, 2.0),
        max_count: int = 4,
        min_steps_between_levels: int = 2000,
        warmup_resets: int = 2,
    ) -> dict[str, float]:
        self._num_calls += 1
        max_count = min(max_count, len(thresholds))

        score = torch.tensor(0.0, device=env.device)
        for term_name in reward_terms:
            score = score + torch.mean(env.reward_manager._episode_sums[term_name][env_ids])
        score = score / env.max_episode_length_s

        has_next_level = self._active_count < max_count
        enough_resets = self._num_calls > warmup_resets
        enough_steps = env.common_step_counter - self._last_increase_step >= min_steps_between_levels
        if has_next_level and enough_resets and enough_steps and score.item() >= thresholds[self._active_count]:
            self._active_count += 1
            self._last_increase_step = env.common_step_counter

        setattr(env, active_count_attr, self._active_count)
        return {"active_count": float(self._active_count), "tracking_score": score}
