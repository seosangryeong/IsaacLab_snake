# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import TerrainBasedPose2dCommandCfg, UniformPose2dCommandCfg,KanakeCommandCfg


class KanakeCommand(CommandTerm):
    """
    pose 커맨드. [x,y,z,heading]
    커맨드는 로봇 위치에서 생성 (샘플링 시 로봇기준으로 (0,0,기본높이)에서 생성)
    """

    cfg: KanakeCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: KanakeCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        self.robot: Articulation = env.scene[cfg.asset_name]

        # create buffers to store the command
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        # metrics
        # self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)

    def _update_metrics(self):
        self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))

    def _resample_command(self, env_ids: Sequence[int]):
        # 로봇의 현재 위치를 (0,0,기본높이)로 간주하여 커맨드 생성
        self.pos_command_w[env_ids, 0] = 0.0
        self.pos_command_w[env_ids, 1] = 0.0
        self.pos_command_w[env_ids, 2] = self.robot.data.default_root_state[env_ids, 2]

        r = torch.empty(len(env_ids), device=self.device)
        self.pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)
        # z는 기본 높이 유지

        if self.cfg.simple_heading:
            # heading은 오프셋 벡터 방향 (로봇이 항상 0방향을 본다고 가정)
            target_vec = self.pos_command_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            curr_heading = torch.zeros_like(target_direction)  # 로봇이 항상 0방향을 본다고 가정
            curr_to_target = wrap_to_pi(target_direction - curr_heading).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - curr_heading).abs()

            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """Re-target the position command to the current root state."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

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

