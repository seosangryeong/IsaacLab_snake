
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

class JointSineAction(ActionTerm):
    """
    [
    
    이 액션 term은 각 환경마다 (조인트 개수 + 4)개의 파라미터를 사용
      - 각 조인트별 진폭 (num_joints)
      - 수직 조인트: frequency_vertical, phase_vertical
      - 수평 조인트: frequency_horizontal, phase_horizontal

      position = amplitude[joint] * sin(2π * frequency * t + (조인트 번호) * phase)
      
    """
    cfg: actions_cfg.JointSineActionCfg
    _asset: Articulation
    _current_time: float

    def __init__(self, cfg: actions_cfg.JointSineActionCfg, env: ManagerBasedEnv) -> None:
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

        self._vertical_joint_names = []
        self._horizontal_joint_names = []
        for name in self._joint_names:
            try:
                number = int(name[1:])
            except Exception as e:
                omni.log.warn(f"조인트 이름 {name} 파싱 실패: {e}")
                number = 0
            if number % 2 == 1:
                self._vertical_joint_names.append(name)
            else:
                self._horizontal_joint_names.append(name)
        
        self._num_vertical = len(self._vertical_joint_names)
        self._num_horizontal = len(self._horizontal_joint_names)

        self._raw_actions = torch.zeros(self.num_envs, self._num_joints + 4, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        self._current_time = 0.0

    @property
    def action_dim(self) -> int:
        return self._num_joints + 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def update_time(self, dt: float):
        self._current_time += dt

    def process_actions(self, actions: torch.Tensor, additional_joint_values: torch.Tensor = None):
        dt = self._env.step_dt
        self.update_time(dt)

        # RL 에이전트의 원본 출력을 raw_actions에 저장
        self._raw_actions[:] = actions


        amp_actions = actions[:, :self._num_joints]
        freq_v_action = actions[:, self._num_joints]
        phase_v_action = actions[:, self._num_joints + 1]
        freq_h_action = actions[:, self._num_joints + 2]
        phase_h_action = actions[:, self._num_joints + 3]



        ########
        amplitudes = 0.5 +(torch.tanh(amp_actions)+ 1.0) / 2.0

        freq_v = 0.7 + 0.7 * (torch.tanh(freq_v_action) + 1.0) / 2.0
        freq_h = 0.7 + 0.7 * (torch.tanh(freq_h_action) + 1.0) / 2.0

        # 위상 (Phase)
        phase_v = np.pi/4 * torch.tanh(phase_v_action) 
        phase_h = np.pi/4 * torch.tanh(phase_h_action) 
        # 진폭 (Amplitude): 0.5 ~ 1.5
        # amplitudes = 0.5 + (torch.nn.functional.softsign(amp_actions) + 1.0) / 2.0

        # # 주파수 (Frequency): 0.3 ~ 1.1
        # freq_v = 0.7 + 0.7 * (torch.nn.functional.softsign(freq_v_action / 5) + 1.0) / 2.0
        # freq_h = 0.7 + 0.7 * (torch.nn.functional.softsign(freq_h_action / 5) + 1.0) / 2.0

        # # 위상 (Phase): -π/4 ~ π/4
        # phase_v = np.pi / 4 * torch.nn.functional.softsign(phase_v_action / 5)
        # phase_h = np.pi / 4 * torch.nn.functional.softsign(phase_h_action / 5)
        

        
        t = torch.full((self.num_envs, 1), self._current_time, device=self.device)

        # 조인트별로 진폭을 나눠서 적용
        vertical_joint_sorted = sorted(self._vertical_joint_names, key=lambda name: int(name[1:]))
        vertical_indices = [self._joint_names.index(name) for name in vertical_joint_sorted]
        # vertical_numbers = torch.arange(len(vertical_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)
        vertical_numbers = torch.arange(len(vertical_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)


        horizontal_joint_sorted = sorted(self._horizontal_joint_names, key=lambda name: int(name[1:]))
        horizontal_indices = [self._joint_names.index(name) for name in horizontal_joint_sorted]
        # horizontal_numbers = torch.arange(len(horizontal_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)
        horizontal_numbers = torch.arange(len(horizontal_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)

        # 각 조인트별 진폭 추출 (스케일링된 amplitudes 사용)
        amp_v = amplitudes[:, vertical_indices] if vertical_indices else torch.zeros(self.num_envs, 0, device=self.device)
        amp_h = amplitudes[:, horizontal_indices] if horizontal_indices else torch.zeros(self.num_envs, 0, device=self.device)

        # 위치 계산 (스케일링된 freq, phase 등 사용)
        vertical_pos = amp_v * torch.sin(
            2 * np.pi * freq_v.unsqueeze(1) * t + vertical_numbers * phase_v.unsqueeze(1)
        ) if vertical_indices else torch.zeros(self.num_envs, 0, device=self.device)
        
        horizontal_pos = amp_h * torch.sin(
            2 * np.pi * freq_h.unsqueeze(1) * t + horizontal_numbers * phase_h.unsqueeze(1)
        ) if horizontal_indices else torch.zeros(self.num_envs, 0, device=self.device)

        # 최종 결과 통합
        processed = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        for i, name in enumerate(self._joint_names):
            if int(name[1:]) % 2 == 1:
                idx = vertical_joint_sorted.index(name)
                processed[:, i] = vertical_pos[:, idx]
            else:
                idx = horizontal_joint_sorted.index(name)
                processed[:, i] = horizontal_pos[:, idx]

        if additional_joint_values is not None:
            processed += self.cfg.additional_joint_scale * additional_joint_values

        processed *= self.cfg.scale

        self._processed_actions.copy_(processed)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0



