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
from isaaclab.utils.math import quat_from_euler_xyz, wrap_to_pi
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import KanakeBaseCommandCfg


class KanakeBaseCommand(CommandTerm):
    """
    pose 커맨드. [x, y, z, heading]
    base frame 기준으로 커맨드가 생성되며 아웃풋 값은 월드좌표
    """

    cfg: KanakeBaseCommandCfg

    def __init__(self, cfg: KanakeBaseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        self.robot: Articulation = env.scene[cfg.asset_name]

        # base frame 기준으로 저장할 커맨드 버퍼
        self.pos_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_b = torch.zeros(self.num_envs, device=self.device)

        # metrics 초기화
        self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "KanakeCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg
    
    # @property
    # def command(self) -> torch.Tensor:
    #     """
    #     base frame 기준으로 생성된 커맨드를 월드좌표로 변환하여 반환.
    #     Shape: (num_envs, 4) → [x, y, z, heading] (월드좌표)
    #     """
    #     # 현재 로봇의 월드 위치/쿼터니언
    #     root_pos = self.robot.data.root_pos_w  # (B, 3)
    #     root_quat = self.robot.data.root_quat_w  # (B, 4)
    #     # base frame 커맨드 → 월드좌표로 변환
    #     des_pos_w, _ = combine_frame_transforms(root_pos, root_quat, self.pos_command_b)
    #     # heading도 월드 기준으로 변환 (base heading + 현재 heading)
    #     des_heading_w = wrap_to_pi(self.heading_command_b + self.robot.data.heading_w)
    #     return torch.cat([des_pos_w, des_heading_w.unsqueeze(1)], dim=1)

    @property
    def command(self) -> torch.Tensor:
        """
        base frame 기준 pose 반환.
        Shape: (num_envs, 4) → [x, y, z, heading]
        """
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        """
        리샘플링 시점에만 호출
        1) 로봇 기준(0,0,기본높이)에서 x,y,z 오프셋을 uniform 샘플링
        2) simple_heading 여부에 따라 heading 결정
        """
        # (1) position offset 샘플링
        r = torch.zeros((len(env_ids), 3), device=self.device)
        # x, y 범위에서 uniform 샘플링
        r[:, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.ranges.pos_x
        )
        r[:, 1] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.ranges.pos_y
        )
        # z는 초기 루트 높이 그대로
        r[:, 2] = self.robot.data.default_root_state[env_ids, 2]
        # base-frame 커맨드로 저장
        self.pos_command_b[env_ids] = r

        # (2) heading 샘플링
        if self.cfg.simple_heading:
            # 오프셋 벡터 방향을 heading으로
            self.heading_command_b[env_ids] = wrap_to_pi(
                torch.atan2(r[:, 1], r[:, 0])
            )
        else:
            # 지정된 범위에서 랜덤
            self.heading_command_b[env_ids] = torch.empty(
                len(env_ids), device=self.device
            ).uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """
        매 프레임 호출되지만, base-frame 커맨드는
        리샘플링 시에만 갱신하도록 의도했으므로 아무 작업도 하지 않습니다.
        """
        pass

    def _update_metrics(self):
        """
        base-frame 명령과 로봇 기준(0,0)의 차이를 오차로 기록
        - error_pos_2d: 목표 위치까지 거리
        - error_heading: 목표 heading까지 각도 차이
        """
        self.metrics["error_pos_2d"] = torch.norm(
            self.pos_command_b[:, :2], dim=1
        )
        self.metrics["error_heading"] = torch.abs(self.heading_command_b)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """
        디버그 시각화: base-frame 목표 좌표를 표시
        """
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_b,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_b),
                torch.zeros_like(self.heading_command_b),
                self.heading_command_b,
            ),
        )
