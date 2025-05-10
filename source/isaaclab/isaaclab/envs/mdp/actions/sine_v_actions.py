from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
import omni.log
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from . import actions_cfg
import isaaclab.sim as sim_utils
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg

class JointSineVerticalAction(ActionTerm):
    """사인파 기반 액션 term.  
    수평 조인트는 사용하지 않고 수직 조인트에만 사인파를 적용하는 액션
    
      - 수직 조인트 (조인트 이름: j1, j3, j5, ...): amplitude, frequency, phase  
    
       position = amplitude * sin(2π * frequency * t + (조인트 번호) * phase)
    """
    
    cfg: actions_cfg.JointSineVerticalActionCfg
    _asset: Articulation
    _current_time: float

    def __init__(self, cfg: actions_cfg.JointSineVerticalActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        omni.log.info(
            f"Resolved joint names for {self.__class__.__name__}: {self._joint_names} [{self._joint_ids}]"
        )

        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = list(range(self._num_joints))

        # 수직 조인트만 필터링 (j1, j3, j5, ...)
        self._vertical_joint_names = [name for name in self._joint_names if int(name[1:]) % 2 != 0]

        # 수직 조인트 ID만 저장
        if isinstance(self._joint_ids, list):
            self._vertical_joint_ids = [self._joint_ids[i] for i, name in enumerate(self._joint_names) if int(name[1:]) % 2 != 0]
        else:
            self._vertical_joint_ids = list(range(self._num_joints))[::2]  # 홀수 인덱스만

        self._num_vertical = len(self._vertical_joint_names)

        # raw action: 각 환경마다 3차원 (amp, freq, phase)
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)

        # processed action: 수직 조인트만 적용
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        self._current_time = 0.0  # 시뮬레이션 시간 초기화

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def update_time(self, dt: float):
        """내부 시뮬레이션 시간을 dt만큼 업데이트"""
        self._current_time += dt

    def process_actions(self, actions: torch.Tensor):
        """액션을 받아서 수직 조인트에 사인파 계산"""
        dt = self._env.step_dt  
        self.update_time(dt)

        # 클리핑 범위 적용
        clip_ranges = getattr(self.cfg, "clip_ranges", [(-1.0, 1.0)] * 3)
        actions_clipped = torch.empty_like(actions)
        for i in range(3):
            actions_clipped[:, i] = torch.clamp(actions[:, i], min=clip_ranges[i][0], max=clip_ranges[i][1])
        actions = actions_clipped
        self._raw_actions[:] = actions

        # 각 환경별 파라미터 분리
        amplitude = actions[:, 0]
        frequency = actions[:, 1]
        phase = actions[:, 2]

        # 현재 시간을 (num_envs, 1) 텐서로 생성
        t = torch.full((self.num_envs, 1), self._current_time, device=self.device)

        # 수직 조인트의 인덱스 정렬
        vertical_joint_sorted = sorted(self._vertical_joint_names, key=lambda name: int(name[1:]))
        vertical_numbers = torch.arange(len(vertical_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)

        # 수직 조인트 위치 계산
        vertical_pos = amplitude.unsqueeze(1) * torch.sin(
            2 * np.pi * frequency.unsqueeze(1) * t + vertical_numbers * phase.unsqueeze(1)
        )

        # 조인트 순서 맞춰서 결과 할당
        processed = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        for i, name in enumerate(self._joint_names):
            if int(name[1:]) % 2 != 0:
                idx = vertical_joint_sorted.index(name)
                processed[:, i] = vertical_pos[:, idx]

        self._processed_actions.copy_(processed)

    def apply_actions(self):
        """계산된 관절 위치를 수직 조인트의 목표 위치로 설정"""
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._vertical_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """환경 리셋 시 raw action 값을 0으로 초기화"""
        if env_ids is None:
            self._raw_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
