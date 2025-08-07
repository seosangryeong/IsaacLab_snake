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
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import KanakeCommandCfg


class KanakeWorldCommand(CommandTerm):
    """
    pose 커맨드. [x, y, z, heading]
    로봇 베이스를 기준으로 커맨드를 생성하고, 아웃풋은 월드 좌표로 반환합니다.
    """

    cfg: KanakeWorldCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: KanakeWorldCommandCfg, env: ManagerBasedEnv):
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
        # 월드 좌표계의 목표 위치/헤딩 (리샘플링 시점에 생성)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)

        # 베이스 좌표계의 상대 위치/헤딩 (매 프레임 업데이트됨)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

        # metrics
        self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "KanakeCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """
        베이스 프레임 기준으로 변환된 커맨드.
        Shape: (num_envs, 4) -> [x, y, z, heading].
        """
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)

    def _update_metrics(self):
        """
        월드 좌표계 목표와 현재 로봇 상태의 오차를 계산합니다.
        """
        self.metrics["error_pos_2d"] = torch.norm(
            self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1
        )
        self.metrics["error_heading"] = torch.abs(
            wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)
        )

    def _resample_command(self, env_ids: Sequence[int]):
            """
            로봇의 베이스 좌표계 기준으로 오프셋을 샘플링한 후,
            이를 월드 좌표로 변환하여 목표를 설정합니다.
            """
            # 1. 로봇 베이스 좌표계 기준 오프셋 샘플링
            # Z 좌표는 항상 0으로 설정하여 로봇의 현재 z 높이를 따라가도록 합니다.
            pos_offset_b = torch.zeros((len(env_ids), 3), device=self.device)
            pos_offset_b[:, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.pos_x)
            pos_offset_b[:, 1] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.pos_y)
            # pos_offset_b[:, 2] = 0.0  # Z 오프셋을 0으로 설정

            # 2. 샘플링된 오프셋을 월드 좌표로 변환하여 목표 위치 설정
            root_pos_w = self.robot.data.root_pos_w[env_ids]
            root_quat_w = self.robot.data.root_quat_w[env_ids]
            
            # combine_frame_transforms를 사용하여 x, y 오프셋만 변환
            pos_command_w_xy, _ = combine_frame_transforms(root_pos_w, root_quat_w, pos_offset_b)
            
            # 월드 좌표 z는 로봇의 현재 z 좌표를 그대로 사용
            self.pos_command_w[env_ids, :2] = pos_command_w_xy[:, :2]
            self.pos_command_w[env_ids, 2] = root_pos_w[:, 2]

            # 3. 헤딩 샘플링 및 월드 좌표로 변환하여 목표 헤딩 설정
            if self.cfg.simple_heading:
                # 오프셋 벡터 방향을 헤딩으로 사용
                target_direction_b = torch.atan2(pos_offset_b[:, 1], pos_offset_b[:, 0])
                self.heading_command_w[env_ids] = wrap_to_pi(target_direction_b + self.robot.data.heading_w[env_ids])
            else:
                # 지정된 범위에서 랜덤 헤딩을 월드 좌표로 변환
                random_heading_b = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.heading)
                self.heading_command_w[env_ids] = wrap_to_pi(random_heading_b + self.robot.data.heading_w[env_ids])
    def _update_command(self):
        """
        매 프레임 호출되어, 월드 좌표계의 목표를 현재 로봇 위치/헤딩 기준의
        베이스 좌표계 커맨드로 재계산합니다.
        """
        # 월드 목표까지 남은 상대 벡터 계산
        target_vec_w = self.pos_command_w - self.robot.data.root_pos_w
        # 이 벡터를 로봇 베이스 좌표계로 변환
        self.pos_command_b[:] = quat_apply_inverse(self.robot.data.root_quat_w, target_vec_w)

        # 월드 목표 heading과 현재 로봇 heading 간의 상대 heading 계산
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
        """
        디버그 시각화: 월드 좌표계의 목표 위치를 표시합니다.
        """
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_w,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
        )