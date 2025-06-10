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
import math
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class JointSineHoldAction(ActionTerm):
    """사인파 기반 액션 term.

    각 환경마다 6개의 파라미터를 사용한다.

      - 수직 조인트 (j1, j3, j5, …) : amplitude_vertical, frequency_vertical, phase_vertical
      - 수평 조인트 (j2, j4, j6, …) : amplitude_horizontal, frequency_horizontal, phase_horizontal

    파라미터는 **최소 0.5 주기, 최대 1 주기** 동안 -또는 파동이
    머리→꼬리(첫-마지막 조인트)까지 전달될 때까지- 고정(hold)된다.
    """

    cfg: actions_cfg.JointSineHoldActionCfg
    _asset: Articulation
    _current_time: float

    # ───────────────────────── 초기화 ──────────────────────────
    def __init__(self, cfg: actions_cfg.JointSineHoldActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        # ── 조인트 식별 ───────────────────────────────────────
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        omni.log.info(
            f"Resolved joint names for {self.__class__.__name__}: {self._joint_names} [{self._joint_ids}]"
        )
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = list(range(self._num_joints))

        self._vertical_joint_names: list[str] = []
        self._horizontal_joint_names: list[str] = []
        for name in self._joint_names:
            try:
                number = int(name[1:])  # 예: j1 -> 1
            except Exception as e:
                omni.log.warn(f"조인트 이름 {name} 실패: {e}")
                number = 0
            (self._vertical_joint_names if number % 2 == 1 else self._horizontal_joint_names).append(name)

        # 조인트 인덱스(홀수/짝수)
        if isinstance(self._joint_ids, list):
            self._even_joint_ids = [self._joint_ids[i] for i, n in enumerate(self._joint_names) if int(n[1:]) % 2 == 1]
            self._odd_joint_ids  = [self._joint_ids[i] for i, n in enumerate(self._joint_names) if int(n[1:]) % 2 == 0]
        else:
            self._even_joint_ids = list(range(self._num_joints))[::2]
            self._odd_joint_ids  = list(range(self._num_joints))[1::2]

        # ── 액션/상태 버퍼 ────────────────────────────────────
        self._raw_actions       = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # **파라미터 홀드**용 버퍼 및 카운터
        self._action_buffer     = torch.zeros_like(self._raw_actions)           # 현재 적용 중인 파라미터
        self._hold_min_steps    = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._hold_max_steps    = torch.zeros_like(self._hold_min_steps)

        # 파동 전파 길이(조인트 수)
        self._N_v = len(self._vertical_joint_names)
        self._N_h = len(self._horizontal_joint_names)

        # 시간
        self._current_time = 0.0

    # ───────────────────────── 속성 ───────────────────────────
    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # ───────────────────────── 헬퍼 ───────────────────────────
    def _compute_hold_steps(self, acts: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """acts : (B,6) → (steps_min, steps_max) [int32]"""

        f_v   = acts[:, 1].abs().clamp_min(1e-3)   # Hz
        f_h   = acts[:, 4].abs().clamp_min(1e-3)
        phi_v = acts[:, 2].abs()                   # rad
        phi_h = acts[:, 5].abs()

        # ① 파동이 꼬리까지 가는 데 걸리는 시간
        t_v = (self._N_v - 1) * phi_v / (2 * math.pi * f_v)
        t_h = (self._N_h - 1) * phi_h / (2 * math.pi * f_h)
        t_wave = torch.maximum(t_v, t_h)

        # ② 느린 쪽 0.5 ↔ 1 주기
        f_min  = torch.minimum(f_v, f_h)
        t_half = self.cfg.min_cycles / f_min
        t_full = self.cfg.max_cycles / f_min

        # ③ 최소/최대 홀드 시간
        t_min = torch.maximum(t_wave, t_half)
        t_max = torch.maximum(t_min,  t_full)

        steps_min = torch.ceil(t_min / dt).to(torch.int32)
        steps_max = torch.ceil(t_max / dt).to(torch.int32)
        return steps_min, steps_max

    def update_time(self, dt: float):
        self._current_time += dt

    # ──────────────────────── 메인 로직 ───────────────────────
    def process_actions(self, actions: torch.Tensor, additional_joint_values: torch.Tensor | None = None):
        dt = self._env.step_dt
        self.update_time(dt)

        # 1) 클리핑 (정책 출력 → actions_clipped)
        clip_ranges = getattr(self.cfg, "clip_ranges", [(-1.0, 1.0)] * 6)
        actions_clipped = torch.empty_like(actions)
        for i in range(6):
            actions_clipped[:, i] = torch.clamp(actions[:, i], *clip_ranges[i])

        # 2) 업데이트 가능 env (최소 홀드 끝난 곳)
        can_update = self._hold_min_steps <= 0
        if can_update.any():
            self._action_buffer[can_update] = actions_clipped[can_update]

            steps_min, steps_max = self._compute_hold_steps(self._action_buffer[can_update], dt)
            self._hold_min_steps[can_update] = steps_min
            self._hold_max_steps[can_update] = steps_max

        # 3) 카운트다운
        self._hold_min_steps -= 1
        self._hold_max_steps -= 1

        # 최대 홀드가 끝났으면 다음 스텝에 무조건 교체되도록
        force_next = self._hold_max_steps <= 0
        self._hold_min_steps[force_next] = 0

        # 4) 이번 스텝에 사용할 파라미터 = 버퍼
        self._raw_actions.copy_(self._action_buffer)

        # ── 파라미터 분해 ─────────────────────────────────────
        amplitude_vertical, frequency_vertical, phase_vertical = \
            self._action_buffer[:, 0], self._action_buffer[:, 1], self._action_buffer[:, 2]
        amplitude_horizontal, frequency_horizontal, phase_horizontal = \
            self._action_buffer[:, 3], self._action_buffer[:, 4], self._action_buffer[:, 5]

        # 시뮬 시간 텐서
        t = torch.full((self.num_envs, 1), self._current_time, device=self.device)

        # 조인트 번호 텐서
        vertical_joint_sorted   = sorted(self._vertical_joint_names,   key=lambda n: int(n[1:]))
        horizontal_joint_sorted = sorted(self._horizontal_joint_names, key=lambda n: int(n[1:]))

        vertical_numbers   = torch.arange(len(vertical_joint_sorted),   device=self.device, dtype=torch.float32).unsqueeze(0)
        horizontal_numbers = torch.arange(len(horizontal_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)

        # 위치 계산
        vertical_pos = amplitude_vertical.unsqueeze(1) * torch.sin(
            2 * np.pi * frequency_vertical.unsqueeze(1) * t + vertical_numbers * phase_vertical.unsqueeze(1)
        )
        horizontal_pos = amplitude_horizontal.unsqueeze(1) * torch.sin(
            2 * np.pi * frequency_horizontal.unsqueeze(1) * t + horizontal_numbers * phase_horizontal.unsqueeze(1)
        )

        # 원래 순서대로 결합
        processed = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        for i, name in enumerate(self._joint_names):
            if int(name[1:]) % 2 == 1:
                idx = vertical_joint_sorted.index(name)
                processed[:, i] = vertical_pos[:, idx]
            else:
                idx = horizontal_joint_sorted.index(name)
                processed[:, i] = horizontal_pos[:, idx]

        # 추가 조인트 입력
        if additional_joint_values is not None:
            processed += self.cfg.additional_joint_scale * additional_joint_values

        self._processed_actions.copy_(processed)

    # ────────────────────── 액션 적용 & 리셋 ───────────────────
    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._action_buffer.zero_()
            self._hold_min_steps.zero_()
            self._hold_max_steps.zero_()
        else:
            self._raw_actions[env_ids]       = 0.0
            self._action_buffer[env_ids]     = 0.0
            self._hold_min_steps[env_ids]    = 0
            self._hold_max_steps[env_ids]    = 0
