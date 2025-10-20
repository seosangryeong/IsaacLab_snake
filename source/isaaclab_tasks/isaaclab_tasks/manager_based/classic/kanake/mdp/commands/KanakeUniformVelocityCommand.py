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

    from .commands_cfg import NormalVelocityCommandCfg,KanakeUniformVelocityCommandCfg


class KanakeUniformVelocityCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: KanakeUniformVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: KanakeUniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
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
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

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
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

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
            self.goal_vel_visualizer.set_visibility(False)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """
        [Kanake 오버라이드]
        시각화의 기준점을 '머리(root)'가 아닌,
        모든 body의 "단순 평균 위치"로 수정합니다.
        
        또한, "현재 속도"도 "단순 평균 속도"를 계산하여 표시합니다.
        """
        if not self.robot.is_initialized:
            return

        # --- 1. "평균 위치" 계산 (시각화 기준점) ---
        # (질량이 거의 동일하다고 가정)
        # (A) 모든 body의 월드 위치를 가져옵니다.
        body_pos_w = self.robot.data.body_com_pos_w[:, :, :3]
        
        # (B) 모든 위치를 더합니다.
        total_pos_w = torch.sum(body_pos_w, dim=1)
        
        # (C) body 개수로 나누어 "평균 위치"를 계산합니다.
        num_bodies = body_pos_w.shape[1]
        avg_pos_w = total_pos_w / num_bodies
        
        # (D) 화살표를 지면에서 0.5m 띄워서 그립니다.
        avg_pos_w[:, 2] += 0.5
        # --- 계산 끝 ---


        # --- 2. "명령" 화살표 (초록색) ---
        #    - _resolve_xy_velocity_to_arrow가 지면에 평행하게 그려줍니다.
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        #    - [수정] 'base_pos_w' 대신 'avg_pos_w'에 그립니다.
        self.goal_vel_visualizer.visualize(avg_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)


        # --- 3. "현재 평균 속도" 화살표 (파란색) ---
        
        # (A) "단순 평균 속도" 계산 (월드 기준)
        body_lin_vels_w = self.robot.data.body_com_vel_w[:, :, :3]
        total_vel_w = torch.sum(body_lin_vels_w, dim=1)
        # (num_bodies는 위에서 이미 계산함)
        current_avg_vel_w = total_vel_w / num_bodies
        
        # (B) "Heading Frame"으로 변환 (지면 투영)
        current_heading_w = self.robot.data.heading_w
        yaw_quat_w = math_utils.quat_from_euler_xyz(
            torch.zeros_like(current_heading_w),
            torch.zeros_like(current_heading_w),
            current_heading_w
        )
        current_avg_vel_heading_frame = math_utils.quat_apply_inverse(
            yaw_quat_w, current_avg_vel_w
        )[:, :2] # XY 성분만 필요

        # (C) 계산된 "평균 속도"를 시각화 함수에 전달
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(current_avg_vel_heading_frame)
        #    - [수정] 'base_pos_w' 대신 'avg_pos_w'에 그립니다.
        self.current_vel_visualizer.visualize(avg_pos_w, vel_arrow_quat, vel_arrow_scale)


    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        [Kanake 오버라이드]
        XY 속도 명령을 화살표의 방향과 크기로 변환합니다.
        (이 함수는 '명령'과 '현재 속도' 시각화 모두에 사용됩니다.)

        원본과 달리, 로봇의 전체 3D 방향(root_quat_w) 대신,
        Roll/Pitch가 0으로 고정된 "Heading(Yaw) 방향(yaw_quat_w)"을 사용합니다.
        """
        # 1. 마커의 기본 스케일 가져오기
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale

        # 2. 화살표 크기 (속도 크기에 비례)
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        # 3. 화살표의 상대적 방향 (로봇 기준)
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat_relative = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        # 4. [핵심 수정] 화살표의 절대(월드) 방향 계산
        #    로봇의 Roll/Pitch를 무시한 "Heading(Yaw) 전용" 쿼터니언을 만듭니다.
        current_heading_w = self.robot.data.heading_w
        yaw_quat_w = math_utils.quat_from_euler_xyz(
            torch.zeros_like(current_heading_w),
            torch.zeros_like(current_heading_w),
            current_heading_w
        )

        # 5. 로봇의 "Heading" 방향에 "상대적 화살표 방향"을 곱합니다.
        arrow_quat_w = math_utils.quat_mul(yaw_quat_w, arrow_quat_relative)

        return arrow_scale, arrow_quat_w
