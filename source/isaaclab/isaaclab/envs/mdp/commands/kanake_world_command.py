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
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat, euler_xyz_from_quat
from isaaclab.utils.math import combine_frame_transforms, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import KanakeCommandCfg


class KanakeWorldCommand(CommandTerm):
    """
    월드 좌표계 기준으로 Z 위치, Yaw, Pitch를 목표로 하는 커맨드 생성기.

    이 클래스는 (x, y)를 제외하고 로봇이 도달해야 할 목표 [z, yaw, pitch]를
    월드 좌표계에서 직접 샘플링하여 제공합니다.
    """

    cfg: KanakeWorldCommandCfg

    def __init__(self, cfg: KanakeWorldCommandCfg, env: ManagerBasedEnv):
        """초기화 함수."""
        super().__init__(cfg, env)

        # 제어할 로봇 에셋 가져오기
        self.robot: Articulation = env.scene[cfg.asset_name]

        # 커맨드 버퍼 생성 (월드 좌표계 기준)
        # command_w의 _w는 world frame을 의미
        self.z_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pitch_command_w = torch.zeros(self.num_envs, device=self.device)
        self.yaw_command_w = torch.zeros(self.num_envs, device=self.device)

        # 메트릭(오차) 버퍼 생성
        self.metrics["error_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """클래스 정보를 문자열로 반환."""
        msg = "WorldPoseCommand:\n"
        msg += f"\tCommand dimension: (3,) -> [z, yaw, pitch]\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """
        목표 커맨드 [z, yaw, pitch]를 반환합니다. (월드 좌표계 기준)
        Shape: (num_envs, 3)
        """
        return torch.stack([self.z_command_w, self.yaw_command_w, self.pitch_command_w], dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        """
        새로운 커맨드를 월드 좌표계에서 샘플링합니다.
        """
        num_resamples = len(env_ids)
        # 설정된 범위 내에서 z, pitch, yaw 값을 균등하게 샘플링
        self.z_command_w[env_ids] = torch.empty(num_resamples, device=self.device).uniform_(*self.cfg.ranges.pos_z)
        self.pitch_command_w[env_ids] = torch.empty(num_resamples, device=self.device).uniform_(*self.cfg.ranges.pitch)
        self.yaw_command_w[env_ids] = torch.empty(num_resamples, device=self.device).uniform_(*self.cfg.ranges.yaw)

    def _update_command(self):

        pass

    def _update_metrics(self):
        """
        목표 커맨드와 로봇의 현재 상태 사이의 오차를 계산합니다.
        """
        # Z 위치 오차
        self.metrics["error_z"] = torch.abs(self.z_command_w - self.robot.data.root_pos_w[:, 2])

        # Pitch 및 Yaw 오차 계산
        # 로봇의 현재 쿼터니언으로부터 오일러 각 (roll, pitch, yaw) 추출
        _, current_pitch, current_yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        # 각도 오차는 wrap_to_pi를 사용해 -pi ~ pi 범위에서 최단 거리를 계산
        self.metrics["error_pitch"] = torch.abs(wrap_to_pi(self.pitch_command_w - current_pitch))
        self.metrics["error_yaw"] = torch.abs(wrap_to_pi(self.yaw_command_w - current_yaw))


    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current head pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """매 시뮬레이션 스텝마다 호출되어 시각화를 업데이트합니다."""
        # 로봇 초기화 확인
        if not self.robot.is_initialized:
            return
        
        # -- 목표 포즈 시각화
        # 현재 로봇 x,y 위치에 목표 z 적용
        translations = self.robot.data.root_pos_w.clone()
        translations[:, 2] = self.z_command_w
        
        # 목표 회전 쿼터니언 생성
        orientations = quat_from_euler_xyz(
            torch.zeros(self.num_envs, device=self.device),
            self.pitch_command_w,
            self.yaw_command_w,
        )
        
        # 목표 포즈 마커 업데이트
        self.goal_pose_visualizer.visualize(
            translations=translations,
            orientations=orientations,
        )
        
        # -- 현재 헤드 포즈 시각화
        try:
            # 헤드 인덱스 찾기
            head_idx = self.robot.body_names.index("head")
            # 헤드 위치와 방향 가져오기
            head_pos = self.robot.data.body_pos_w[:, head_idx]
            head_quat = self.robot.data.body_quat_w[:, head_idx]
            # 헤드 프레임 마커 업데이트
            self.current_pose_visualizer.visualize(
                translations=head_pos,
                orientations=head_quat,
            )
        except (ValueError, IndexError) as e:
            # 헤드 바디를 찾을 수 없는 경우 처리
            print(f"Warning: Could not visualize head frame: {e}")