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

    from .commands_cfg import KanakeWorldCommandCfg


class KanakeWorldCommand(CommandTerm):
    """
    월드 좌표계 기준으로 Z 위치, Roll, Yaw, Pitch를 목표로 하는 커맨드 

    [z, roll, yaw, pitch]를 월드 좌표계에서 직접 샘플링
    """

    cfg: KanakeWorldCommandCfg

    def __init__(self, cfg: KanakeWorldCommandCfg, env: ManagerBasedEnv):
        """초기화 함수."""
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]


        self.z_command_w = torch.zeros(self.num_envs, device=self.device)
        self.roll_command_w = torch.zeros(self.num_envs, device=self.device) 
        self.pitch_command_w = torch.zeros(self.num_envs, device=self.device)
        self.yaw_command_w = torch.zeros(self.num_envs, device=self.device)

        self.z_command_b = torch.zeros(self.num_envs, device=self.device)
        self.roll_command_b = torch.zeros(self.num_envs, device=self.device)
        self.pitch_command_b = torch.zeros(self.num_envs, device=self.device)
        self.yaw_command_b = torch.zeros(self.num_envs, device=self.device)

        self.metrics["error_z"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_roll"] = torch.zeros(self.num_envs, device=self.device)  
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_yaw"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "WorldPoseCommand:\n"
        msg += f"\tCommand dimension: (4,) -> [z, roll, yaw, pitch]\n"  
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """
        목표 커맨드 [z, roll, yaw, pitch]를 반환
        Shape: (num_envs, 4)
        """
        return torch.stack([self.z_command_b, self.roll_command_b, self.yaw_command_b, self.pitch_command_b], dim=1)

    def _resample_command(self, env_ids: Sequence[int]):

        num_resamples = len(env_ids)
        self.z_command_w[env_ids] = torch.empty(num_resamples, device=self.device).uniform_(*self.cfg.ranges.pos_z)
        self.roll_command_w[env_ids] = torch.empty(num_resamples, device=self.device).uniform_(*self.cfg.ranges.roll)  # Roll 샘플링 추가
        self.pitch_command_w[env_ids] = torch.empty(num_resamples, device=self.device).uniform_(*self.cfg.ranges.pitch)
        self.yaw_command_w[env_ids] = torch.empty(num_resamples, device=self.device).uniform_(*self.cfg.ranges.yaw)

    def _update_command(self):
        self.z_command_b[:] = self.z_command_w - self.robot.data.root_pos_w[:, 2]

        # Roll, Pitch, Yaw 오차 계산
        current_roll, current_pitch, current_yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)

        self.roll_command_b[:] = wrap_to_pi(self.roll_command_w - current_roll)
        self.pitch_command_b[:] = wrap_to_pi(self.pitch_command_w - current_pitch)
        self.yaw_command_b[:] = wrap_to_pi(self.yaw_command_w - current_yaw)

    def _update_metrics(self):

        cube_idx = self.robot.body_names.index("cube")
        cube_z = self.robot.data.body_pos_w[:, cube_idx, 2]
        self.metrics["error_z"] = torch.abs(self.z_command_w - cube_z)


        current_roll, current_pitch, current_yaw = euler_xyz_from_quat(self.robot.data.root_quat_w)
        self.metrics["error_roll"] = torch.abs(wrap_to_pi(self.roll_command_w - current_roll))
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
        if not self.robot.is_initialized:
            return
        
        # 현재 로봇 x,y 위치에 목표 z 적용
        translations = self.robot.data.root_pos_w.clone()
        translations[:, 2] = self.z_command_w
        
        # 목표 회전 쿼터니언 생성 
        orientations = quat_from_euler_xyz(
            self.roll_command_w, 
            self.pitch_command_w,
            self.yaw_command_w,
        )
        
        self.goal_pose_visualizer.visualize(
            translations=translations,
            orientations=orientations,
        )
        
        try:
            head_idx = self.robot.body_names.index("cube")  
            head_pos = self.robot.data.body_pos_w[:, head_idx]
            head_quat = self.robot.data.body_quat_w[:, head_idx]
            self.current_pose_visualizer.visualize(
                translations=head_pos,
                orientations=head_quat,
            )
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not visualize head frame: {e}")