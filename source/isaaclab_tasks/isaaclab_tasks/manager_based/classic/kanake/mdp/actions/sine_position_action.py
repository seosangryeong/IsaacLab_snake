from __future__ import annotations
import numpy as np
import torch
from collections.abc import Sequence
import omni.log

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from . import actions_cfg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class JointSinePositionAction(ActionTerm):
    """사인파 기반 + 상대 위치 offset 액션 term.
    
    - 액션 차원: 6개(사인파) + N개(offset) = 6 + num_joints
    - position = sine_wave + offset
    """

    cfg: actions_cfg.JointSinePositionActionCfg
    _asset: Articulation
    _current_time: float

    def __init__(self, cfg: actions_cfg.JointSinePositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        # 조인트 찾기
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        omni.log.info(f"Resolved joint names for {self.__class__.__name__}: {self._joint_names} [{self._joint_ids}]")

        # 수직 / 수평 구분
        self._vertical_joint_names = []
        self._horizontal_joint_names = []
        for name in self._joint_names:
            try:
                number = int(name[1:])
                if number % 2 == 1:
                    self._vertical_joint_names.append(name)
                else:
                    self._horizontal_joint_names.append(name)
            except Exception as e:
                omni.log.warn(f"조인트 이름 파싱 실패: {name}, {e}")

        # 사인파 + offset 포함 전체 액션
        self._raw_actions = torch.zeros(self.num_envs, 6 + self._num_joints, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        self._current_time = 0.0

    @property
    def action_dim(self) -> int:
        return 6 + self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def update_time(self, dt: float):
        self._current_time += dt

    def process_actions(self, actions: torch.Tensor):
        dt = self._env.step_dt
        self.update_time(dt)

        # clip range 
        default_offset_clip = (-0.2, 0.2)
        clip_ranges = getattr(self.cfg, "clip", [(-1.0, 1.0)] * self.action_dim)
        if len(clip_ranges) < self.action_dim:
            extra_clips = [default_offset_clip] * (self.action_dim - len(clip_ranges))
            clip_ranges = clip_ranges + extra_clips

        # 클리핑 적용
        actions_clipped = torch.empty_like(actions)
        for i in range(self.action_dim):
            actions_clipped[:, i] = torch.clamp(actions[:, i], min=clip_ranges[i][0], max=clip_ranges[i][1])
        actions = actions_clipped
        self._raw_actions[:] = actions

        # 사인파 파라미터
        amplitude_v = actions[:, 0]
        freq_v     = actions[:, 1]
        phase_v    = actions[:, 2]
        amplitude_h = actions[:, 3]
        freq_h      = actions[:, 4]
        phase_h     = actions[:, 5]

        # offset 값
        offsets = actions[:, 6:]  # shape: [num_envs, num_joints]

        t = torch.full((self.num_envs, 1), self._current_time, device=self.device)

        vertical_sorted = sorted(self._vertical_joint_names, key=lambda name: int(name[1:]))
        vertical_idx = torch.arange(len(vertical_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)

        horizontal_sorted = sorted(self._horizontal_joint_names, key=lambda name: int(name[1:]))
        horizontal_idx = torch.arange(len(horizontal_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)

        # 사인파 계산
        pos_v = amplitude_v.unsqueeze(1) * torch.sin(2 * np.pi * freq_v.unsqueeze(1) * t + vertical_idx * phase_v.unsqueeze(1))
        pos_h = amplitude_h.unsqueeze(1) * torch.sin(2 * np.pi * freq_h.unsqueeze(1) * t + horizontal_idx * phase_h.unsqueeze(1))

        # 조인트 순서대로 재구성 + offset 추가
        final = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        for i, name in enumerate(self._joint_names):
            n = int(name[1:])
            if n % 2 == 1:
                idx = vertical_sorted.index(name)
                final[:, i] = pos_v[:, idx]
            else:
                idx = horizontal_sorted.index(name)
                final[:, i] = pos_h[:, idx]
        
        final += offsets  # offset 적용
        self._processed_actions.copy_(final)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
