# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_from_euler_xyz, wrap_to_pi, yaw_quat
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import KanakeCommandCfg


class KanakeCommand(CommandTerm):
    """
    pose 커맨드. [x, y, z, heading]
    커맨드는 월드 좌표계 기준으로 직접 샘플링 (로봇 위치와 무관하게 고정)
    """

    cfg: KanakeCommandCfg

    def __init__(self, cfg: KanakeCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # 월드 좌표계 기준 목표
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)

        self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "KanakeCommand:\n"
        msg += f"\tCommand dimension: (4,)\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """
        월드 기준 목표 [x, y, z, heading]
        Shape: (num_envs, 4)
        """
        return torch.cat([self.pos_command_w, self.heading_command_w.unsqueeze(1)], dim=1)

    def _update_metrics(self):
        self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))

    def _resample_command(self, env_ids: Sequence[int]):

        num_resamples = len(env_ids)
        if num_resamples == 0:
            return

        self.pos_command_w[env_ids, 0] = torch.empty(num_resamples, device=self.device).uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_w[env_ids, 1] = torch.empty(num_resamples, device=self.device).uniform_(*self.cfg.ranges.pos_y)
        self.pos_command_w[env_ids, 2] = torch.empty(num_resamples, device=self.device).uniform_(*self.cfg.ranges.pos_z)
        self.heading_command_w[env_ids] = torch.empty(num_resamples, device=self.device).uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_w,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
        )