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
    from .commands_cfg import TerrainBasedPose2dCommandCfg, UniformPose2dCommandCfg, KanakeWorldCommandCfg


class KanakeWorldCommand(CommandTerm):
    """
    world-coordinate 기반 pose 커맨드. [x, y, z, heading]
    커맨드는 월드 좌표계에서 직접 샘플링.
    """

    cfg: KanakeWorldCommandCfg

    def __init__(self, cfg: KanakeWorldCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # 로봇 아티큘레이션 핸들
        self.robot: Articulation = env.scene[cfg.asset_name]

        # 절대 월드 좌표계 버퍼
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)

        # 메트릭 초기화
        self.metrics["error_pos_2d"]   = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in world frame. Shape is (num_envs, 4)."""
        return torch.cat([self.pos_command_w, self.heading_command_w.unsqueeze(1)], dim=1)

    def _update_metrics(self):
        # world command와 실제 로봇 위치 차이
        self.metrics["error_pos_2d"] = torch.norm(
            self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1
        )
        self.metrics["error_heading"] = torch.abs(
            wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # 절대 월드 좌표계에서 직접 샘플링
        r = torch.empty(len(env_ids), device=self.device)
        self.pos_command_w[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_w[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        # z 축(높이)은 기본 root 높이를 유지
        self.pos_command_w[env_ids, 2] = self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            # world 좌표 기준 heading 계산
            target_vec = self.pos_command_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped = wrap_to_pi(target_direction + torch.pi)

            curr_heading = torch.zeros_like(target_direction)
            to_target = wrap_to_pi(target_direction - curr_heading).abs()
            to_flipped = wrap_to_pi(flipped - curr_heading).abs()

            self.heading_command_w[env_ids] = torch.where(
                to_target < to_flipped,
                target_direction,
                flipped,
            )
        else:
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        # world coordinates를 그대로 사용하므로 변환 불필요
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
