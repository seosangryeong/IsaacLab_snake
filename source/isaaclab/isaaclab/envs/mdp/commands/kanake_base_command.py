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
    from .commands_cfg import KanakeBaseCommandCfg


class KanakeBaseCommand(CommandTerm):
    """
    pose 커맨드. [x,y,z,heading]
    커맨드는 로봇 위치에서 생성 (샘플링 시 로봇기준으로 (0,0,기본높이)에서 생성)
    """

    cfg: KanakeBaseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: KanakeBaseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class."""
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        # Buffers for world-frame targets (fixed between resampling)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        # Buffers for base-frame commands (updated every step)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        
        self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in base frame for the policy.
        
        Shape is (num_envs, 3), corresponding to [x_relative, y_relative, heading_relative].
        """
        pos_command_b_2d = self.pos_command_b[:, :2]
        return torch.cat([pos_command_b_2d, self.heading_command_b.unsqueeze(1)], dim=1)

    def _update_metrics(self):
        """Computes the 2D error between the desired command and the current robot state."""
        self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))

    def _resample_command(self, env_ids: Sequence[int]):
        """Resamples the command relative to the robot's current position."""
        num_resamples = len(env_ids)
        if num_resamples == 0:
            return

        current_robot_pos_w = self.robot.data.root_pos_w[env_ids]
        
        rand_pos_x = torch.empty(num_resamples, device=self.device).uniform_(*self.cfg.ranges.pos_x)
        rand_pos_y = torch.empty(num_resamples, device=self.device).uniform_(*self.cfg.ranges.pos_y)

        self.pos_command_w[env_ids, 0] = current_robot_pos_w[:, 0] + rand_pos_x
        self.pos_command_w[env_ids, 1] = current_robot_pos_w[:, 1] + rand_pos_y
        self.pos_command_w[env_ids, 2] = self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            current_robot_heading_w = self.robot.data.heading_w[env_ids]
            target_vec_w = self.pos_command_w[env_ids, :2] - current_robot_pos_w[:, :2]
            target_direction = torch.atan2(target_vec_w[:, 1], target_vec_w[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)
            curr_to_target = wrap_to_pi(target_direction - current_robot_heading_w).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - current_robot_heading_w).abs()
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target, target_direction, flipped_target_direction
            )
        else:
            r_heading = torch.empty(num_resamples, device=self.device)
            self.heading_command_w[env_ids] = r_heading.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """Re-target the position command to the current root state."""
        target_vec_w = self.pos_command_w - self.robot.data.root_pos_w
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec_w)
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