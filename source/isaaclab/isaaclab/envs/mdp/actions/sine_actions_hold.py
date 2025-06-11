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

class JointSineHoldAction(ActionTerm):
    """사인파 기반 액션 term.
    
    이 액션 term은 각 환경마다 6개의 파라미터를 사용
    
      - 수직 조인트 (j1, j3, j5, ...): amplitude_vertical, frequency_vertical, phase_vertical  
      - 수평 조인트 (j2, j4, j6, ...): amplitude_horizontal, frequency_horizontal, phase_horizontal
    
    position = amplitude * sin(2π * frequency * t + (조인트 번호) * phase)
    
    한 번 설정된 파라미터는 최소 한 주기 동안 유지됩니다.
    """
    cfg: actions_cfg.JointSineActionCfg
    _asset: Articulation
    _current_time: float

    def __init__(self, cfg: actions_cfg.JointSineHoldActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        # 조인트 ID 및 이름 조회
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        omni.log.info(f"Resolved joint names for {self.__class__.__name__}: {self._joint_names} [{self._joint_ids}]")

        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = list(range(self._num_joints))

        # 수직/수평 조인트 분류
        self._vertical_joint_names = [n for n in self._joint_names if int(n[1:]) % 2 == 1]
        self._horizontal_joint_names = [n for n in self._joint_names if int(n[1:]) % 2 == 0]
        self._num_even = len(self._vertical_joint_names)
        self._num_odd = len(self._horizontal_joint_names)

        # raw action: [amp_v, freq_v, phase_v, amp_h, freq_h, phase_h]
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        # 마지막으로 확정된 raw action (한 주기 유지용)
        self._last_raw_actions = self._raw_actions.clone()
        # env별로 마지막 파라미터가 적용된 시뮬레이션 시간 기록
        self._last_update_time = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # processed action: 각 관절 목표 위치
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._current_time = 0.0

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor, additional_joint_values: torch.Tensor = None):
        # 1) 시간 업데이트
        dt = self._env.step_dt
        self._current_time += dt

        # 2) 입력 파라미터 클리핑
        clip_ranges = getattr(self.cfg, "clip_ranges", [(-1.0, 1.0)] * 6)
        actions_clipped = torch.empty_like(actions)
        for i in range(6):
            actions_clipped[:, i] = torch.clamp(actions[:, i],
                                                min=clip_ranges[i][0],
                                                max=clip_ranges[i][1])
        new_actions = actions_clipped

        # 3) 최소 한 주기 유지 로직
        freq_v_last = self._last_raw_actions[:, 1]
        freq_h_last = self._last_raw_actions[:, 4]
        # 주기 계산 (freq > 0 인 경우에만)
        period_v = torch.where(freq_v_last > 0, 1.0 / freq_v_last, torch.zeros_like(freq_v_last))
        period_h = torch.where(freq_h_last > 0, 1.0 / freq_h_last, torch.zeros_like(freq_h_last))
        # 두 주기 중 더 긴 것만큼 유지
        min_hold = torch.max(period_v, period_h)

        delta = self._current_time - self._last_update_time  # shape: (num_envs,)
        need_update = delta >= min_hold  # bool tensor

        # 조건 충족 env만 raw action 갱신
        self._raw_actions[need_update] = new_actions[need_update]
        # 업데이트 시점 기록
        self._last_update_time[need_update] = self._current_time
        self._last_raw_actions[need_update] = self._raw_actions[need_update]

        # 4) 사인파 위치 계산
        amp_v, freq_v, phase_v = self._raw_actions[:, 0], self._raw_actions[:, 1], self._raw_actions[:, 2]
        amp_h, freq_h, phase_h = self._raw_actions[:, 3], self._raw_actions[:, 4], self._raw_actions[:, 5]
        t = torch.full((self.num_envs, 1), self._current_time, device=self.device)

        # 조인트 번호별 정렬
        v_sorted = sorted(self._vertical_joint_names, key=lambda n: int(n[1:]))
        h_sorted = sorted(self._horizontal_joint_names, key=lambda n: int(n[1:]))
        v_idx = torch.arange(len(v_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)
        h_idx = torch.arange(len(h_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)

        v_pos = amp_v.unsqueeze(1) * torch.sin(2*np.pi*freq_v.unsqueeze(1)*t + v_idx*phase_v.unsqueeze(1))
        h_pos = amp_h.unsqueeze(1) * torch.sin(2*np.pi*freq_h.unsqueeze(1)*t + h_idx*phase_h.unsqueeze(1))

        # 원래 순서대로 합치기
        processed = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        for i, name in enumerate(self._joint_names):
            if int(name[1:]) % 2 == 1:
                idx = v_sorted.index(name)
                processed[:, i] = v_pos[:, idx]
            else:
                idx = h_sorted.index(name)
                processed[:, i] = h_pos[:, idx]

        # 추가 joint 값 스케일 적용
        if additional_joint_values is not None:
            processed += self.cfg.additional_joint_scale * additional_joint_values

        self._processed_actions.copy_(processed)

    def apply_actions(self):
        """계산된 관절 위치를 목표 위치로 설정"""
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """환경 리셋 시 raw action 초기화 및 타이머 리셋"""
        if env_ids is None:
            # 전체 env
            self._raw_actions.zero_()
            self._last_raw_actions.zero_()
            self._last_update_time = torch.full(
                (self.num_envs,), self._current_time, device=self.device
            )
        else:
            # 일부 env
            self._raw_actions[env_ids] = 0.0
            self._last_raw_actions[env_ids] = 0.0
            self._last_update_time[env_ids] = self._current_time
