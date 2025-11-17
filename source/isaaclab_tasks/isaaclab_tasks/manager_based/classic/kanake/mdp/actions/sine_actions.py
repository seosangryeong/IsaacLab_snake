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
    이 액션 term은 각 환경마다 (조인트 개수 + 4)개의 파라미터를 사용
      - 각 조인트별 진폭 ΔA_j (num_joints)
      - 수직 조인트: Δfrequency_vertical, Δphase_vertical
      - 수평 조인트: Δfrequency_horizontal, Δphase_horizontal

    실제로 사용하는 파라미터는 이 클래스 안에서 누적:
      - amplitude_state[joint]
      - freq_v_state, phase_v_state
      - freq_h_state, phase_h_state

    position = amplitude_state[joint] * sin(2π * freq_state * t + (조인트 번호) * phase_state)
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

        # 수직/수평 조인트 분리
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

        # 인덱스/번호는 매 step마다 만들 필요 없으니 한 번만 계산
        self._vertical_joint_sorted = sorted(self._vertical_joint_names, key=lambda name: int(name[1:]))
        self._vertical_indices = [self._joint_names.index(name) for name in self._vertical_joint_sorted]
        self._vertical_numbers = torch.arange(
            len(self._vertical_joint_sorted), device=self.device, dtype=torch.float32
        ).unsqueeze(0)

        self._horizontal_joint_sorted = sorted(self._horizontal_joint_names, key=lambda name: int(name[1:]))
        self._horizontal_indices = [self._joint_names.index(name) for name in self._horizontal_joint_sorted]
        self._horizontal_numbers = torch.arange(
            len(self._horizontal_joint_sorted), device=self.device, dtype=torch.float32
        ).unsqueeze(0)

        # RL 원본 출력(action) 저장용
        self._raw_actions = torch.zeros(self.num_envs, self._num_joints + 4, device=self.device)
        # 최종 joint position target
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # 파형 상태(누적 파라미터)
        # amplitude_state: [num_envs, num_joints]
        # freq/phase_state: [num_envs, 1] (vert/horiz 따로)
        self._init_amplitude = 1.0
        self._init_freq_v = 1.4
        self._init_freq_h = 1.4
        self._init_phase_v = np.pi / 2.0
        self._init_phase_h = np.pi / 2.0

        self._amplitude_state = torch.full(
            (self.num_envs, self._num_joints), self._init_amplitude, device=self.device
        )
        self._freq_v_state = torch.full((self.num_envs, 1), self._init_freq_v, device=self.device)
        self._freq_h_state = torch.full((self.num_envs, 1), self._init_freq_h, device=self.device)
        self._phase_v_state = torch.full((self.num_envs, 1), self._init_phase_v, device=self.device)
        self._phase_h_state = torch.full((self.num_envs, 1), self._init_phase_h, device=self.device)

        # Δ 적용 스케일 (얼마나 천천히 바꿀지)
        self._alpha_amp = 0.1    # Δamp 스케일
        self._alpha_freq = 0.02   # Δfreq 스케일
        self._alpha_phase = 0.04  # Δphase 스케일

        # 파라미터 클램프 범위
        self._amp_min = -1.0
        self._amp_max = 1.0
        self._freq_min = 0.1
        self._freq_max = 2.0
        # phase는 -π ~ π 범위에 wrap

        self._current_time = 0.0

    @property
    def action_dim(self) -> int:
        # Δamp_j (num_joints) + Δfreq_v + Δphase_v + Δfreq_h + Δphase_h
        return self._num_joints + 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def update_time(self, dt: float):
        self._current_time += dt

    def _wrap_phase(self, phase: torch.Tensor) -> torch.Tensor:
        # [-π, π] 범위로 wrap
        return (phase + torch.pi) % (2 * torch.pi) - torch.pi

    def process_actions(self, actions: torch.Tensor, additional_joint_values: torch.Tensor = None):
        dt = self._env.step_dt
        self.update_time(dt)

        # RL 에이전트의 원본 출력을 raw_actions에 저장
        self._raw_actions[:] = actions

        # actions: [num_envs, num_joints + 4]
        # 앞 num_joints: Δamplitude_j
        d_amp = actions[:, :self._num_joints]

        # 나머지 4개: Δfreq_v, Δphase_v, Δfreq_h, Δphase_h
        d_freq_v = actions[:, self._num_joints + 0].unsqueeze(-1)   # [num_envs, 1]
        d_phase_v = actions[:, self._num_joints + 1].unsqueeze(-1)
        d_freq_h = actions[:, self._num_joints + 2].unsqueeze(-1)
        d_phase_h = actions[:, self._num_joints + 3].unsqueeze(-1)

        # ----- Δ를 누적해서 "현재 파형 파라미터" 업데이트 -----

        # 진폭 상태 업데이트 + 클램프
        self._amplitude_state = torch.clamp(
            self._amplitude_state + self._alpha_amp * d_amp,
            min=self._amp_min,
            max=self._amp_max,
        )

        # 주파수 상태 업데이트 + 클램프
        self._freq_v_state = torch.clamp(
            self._freq_v_state + self._alpha_freq * d_freq_v,
            min=self._freq_min,
            max=self._freq_max,
        )
        self._freq_h_state = torch.clamp(
            self._freq_h_state + self._alpha_freq * d_freq_h,
            min=self._freq_min,
            max=self._freq_max,
        )

        # 위상 상태 업데이트 + wrap
        self._phase_v_state = self._wrap_phase(self._phase_v_state + self._alpha_phase * d_phase_v)
        self._phase_h_state = self._wrap_phase(self._phase_h_state + self._alpha_phase * d_phase_h)

        # ----- 업데이트된 상태로 파형 계산 -----

        t = torch.full((self.num_envs, 1), self._current_time, device=self.device)

        # 각 조인트별 진폭 추출
        if self._vertical_indices:
            amp_v = self._amplitude_state[:, self._vertical_indices]
        else:
            amp_v = torch.zeros(self.num_envs, 0, device=self.device)

        if self._horizontal_indices:
            amp_h = self._amplitude_state[:, self._horizontal_indices]
        else:
            amp_h = torch.zeros(self.num_envs, 0, device=self.device)

        # 수직 조인트 위치
        if self._vertical_indices:
            vertical_pos = amp_v * torch.sin(
                2 * np.pi * self._freq_v_state * t
                + self._vertical_numbers * self._phase_v_state
            )
        else:
            vertical_pos = torch.zeros(self.num_envs, 0, device=self.device)

        # 수평 조인트 위치
        if self._horizontal_indices:
            horizontal_pos = amp_h * torch.sin(
                2 * np.pi * self._freq_h_state * t
                + self._horizontal_numbers * self._phase_h_state
            )
        else:
            horizontal_pos = torch.zeros(self.num_envs, 0, device=self.device)

        # 최종 결과 통합 (joint 이름 순서대로 다시 배치)
        processed = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        for i, name in enumerate(self._joint_names):
            joint_num = int(name[1:])
            if joint_num % 2 == 1:
                # vertical
                idx = self._vertical_joint_sorted.index(name)
                processed[:, i] = vertical_pos[:, idx]
            else:
                # horizontal
                idx = self._horizontal_joint_sorted.index(name)
                processed[:, i] = horizontal_pos[:, idx]

        if additional_joint_values is not None:
            processed += self.cfg.additional_joint_scale * additional_joint_values

        processed *= self.cfg.scale

        self._processed_actions.copy_(processed)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # env reset 시 파형 상태도 함께 초기화
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            # 전체 env에 대해 상태 초기화
            self._amplitude_state.fill_(self._init_amplitude)
            self._freq_v_state.fill_(self._init_freq_v)
            self._freq_h_state.fill_(self._init_freq_h)
            self._phase_v_state.fill_(self._init_phase_v)
            self._phase_h_state.fill_(self._init_phase_h)
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._amplitude_state[env_ids] = self._init_amplitude
            self._freq_v_state[env_ids] = self._init_freq_v
            self._freq_h_state[env_ids] = self._init_freq_h
            self._phase_v_state[env_ids] = self._init_phase_v
            self._phase_h_state[env_ids] = self._init_phase_h
