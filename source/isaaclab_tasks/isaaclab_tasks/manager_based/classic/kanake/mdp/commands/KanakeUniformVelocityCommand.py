# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import NormalVelocityCommandCfg, KanakeUniformVelocityCommandCfg


class KanakeUniformVelocityCommand(CommandTerm):



    cfg: KanakeUniformVelocityCommandCfg

    def __init__(self, cfg: KanakeUniformVelocityCommandCfg, env: ManagerBasedEnv):

        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        
        # 리샘플링 시점의 고정된 위치/방향 저장
        self.command_spawn_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_spawn_heading_w = torch.zeros(self.num_envs, device=self.device)
        
        # 월드 기준 고정 커맨드 저장 (리샘플링 시점에 생성)
        self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        # """
        # [Kanake 오버라이드]
        # 리샘플링 시점에 생성된 "월드 고정 커맨드"를 현재 로봇의 base frame으로 변환하여 반환.
        
        # Returns:
        #     torch.Tensor: (num_envs, 3) - base frame 기준 속도 커맨드 [vx, vy, w_z]
        # """
        # # 현재 로봇의 heading
        # current_heading_w = self.robot.data.heading_w
        
        # # 월드 → base frame 변환을 위한 quaternion
        # yaw_quat_w = math_utils.quat_from_euler_xyz(
        #     torch.zeros_like(current_heading_w),
        #     torch.zeros_like(current_heading_w),
        #     current_heading_w
        # )
        
        # # 선형 속도 변환 (XY만, Z는 0)
        # vel_command_w_3d = torch.cat([
        #     self.vel_command_w[:, :2],
        #     torch.zeros(self.num_envs, 1, device=self.device)
        # ], dim=1)
        
        # vel_command_b_3d = math_utils.quat_apply_inverse(yaw_quat_w, vel_command_w_3d)
        
        # # base frame 커맨드 업데이트
        # self.vel_command_b[:, :2] = vel_command_b_3d[:, :2]
        # self.vel_command_b[:, 2] = self.vel_command_w[:, 2]  # 각속도는 동일
        
        # return self.vel_command_b
        return self.vel_command_w  

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        """
        리샘플링 시 월드 고정 커맨드 생성
        
        1. 현재 위치/heading 저장
        2. base frame 기준 임시 커맨드 생성
        3. 월드 frame으로 변환하여 저장
        """
        r = torch.empty(len(env_ids), device=self.device)
        
        # 리샘플링 시점의 위치/heading 저장
        body_pos_w = self.robot.data.body_com_pos_w[env_ids, :, :3]
        self.command_spawn_pos_w[env_ids] = torch.mean(body_pos_w, dim=1)
        self.command_spawn_heading_w[env_ids] = self.robot.data.heading_w[env_ids]
        
        # 각 변수마다 새로운 텐서를 생성하여 할당합니다.
        temp_cmd_x = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.lin_vel_x)
        temp_cmd_y = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.lin_vel_y)
        temp_cmd_yaw = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.ang_vel_z)
        
        # base → world frame 변환 (리샘플링 시점의 heading 사용)
        spawn_heading = self.command_spawn_heading_w[env_ids]
        yaw_quat_w = math_utils.quat_from_euler_xyz(
            torch.zeros_like(spawn_heading),
            torch.zeros_like(spawn_heading),
            spawn_heading
        )
        
        temp_cmd_3d = torch.stack([temp_cmd_x, temp_cmd_y, torch.zeros_like(temp_cmd_x)], dim=1)
        vel_cmd_w_3d = math_utils.quat_apply(yaw_quat_w, temp_cmd_3d)
        
        # 월드 고정 커맨드 저장
        self.vel_command_w[env_ids, 0] = vel_cmd_w_3d[:, 0]
        self.vel_command_w[env_ids, 1] = vel_cmd_w_3d[:, 1]
        self.vel_command_w[env_ids, 2] = temp_cmd_yaw  # yaw는 월드/base 동일
        
        # heading 커맨드 처리
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        
        # standing 환경
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):

        # Heading control
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_w[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        
        # Standing
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_w[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)  
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """
        [Kanake 오버라이드 v4 - 최종]
        
        1. 목표(초록색) 화살표: 리샘플링 시점의 위치에 고정 (월드 커맨드 표시)
        2. 현재(파란색) 화살표: 현재 로봇의 평균 위치를 따라감 (월드 속도 표시)
        """
        if not self.robot.is_initialized:
            return

        # --- 1. 현재 로봇의 평균 위치 (파란색 화살표용) ---
        body_pos_w = self.robot.data.body_com_pos_w[:, :, :3]
        current_avg_pos_w = torch.mean(body_pos_w, dim=1)
        current_avg_pos_w[:, 2] += 0.5  # 0.5m 띄우기


        # --- 2. "목표" 화살표 (초록색) - 고정 위치 + 월드 커맨드 ---
        fixed_pos_w = self.command_spawn_pos_w.clone()
        fixed_pos_w[:, 2] += 0.5  # 0.5m 띄우기
        
        # 🔧 월드 커맨드를 직접 시각화 (heading 변환 없이)
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_world_velocity_to_arrow(
            self.vel_command_w[:, :2]
        )
        
        self.goal_vel_visualizer.visualize(fixed_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)


        # --- 3. "현재" 화살표 (파란색) - 현재 위치 + 월드 속도 ---
        body_lin_vels_w = self.robot.data.body_com_vel_w[:, :, :3]
        current_avg_vel_w = torch.mean(body_lin_vels_w, dim=1)
        
        # 월드 속도 시각화
        vel_arrow_scale, vel_arrow_quat = self._resolve_world_velocity_to_arrow(
            current_avg_vel_w[:, :2]
        )
        
        self.current_vel_visualizer.visualize(current_avg_pos_w, vel_arrow_quat, vel_arrow_scale)


    def _resolve_world_velocity_to_arrow(
        self, 
        xy_velocity_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        [새 함수] 월드 기준 XY 속도를 화살표로 변환
        
        Args:
            xy_velocity_w: (num_envs, 2) - 월드 기준 XY 속도
        
        Returns:
            arrow_scale: (num_envs, 3) - 화살표 크기
            arrow_quat_w: (num_envs, 4) - 화살표 방향 (월드 기준)
        """
        # 1. 화살표 크기
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity_w.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity_w, dim=1) * 3.0

        # 2. 월드 기준 화살표 방향 (atan2로 yaw 계산)
        heading_angle_w = torch.atan2(xy_velocity_w[:, 1], xy_velocity_w[:, 0])
        zeros = torch.zeros_like(heading_angle_w)
        arrow_quat_w = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle_w)

        return arrow_scale, arrow_quat_w