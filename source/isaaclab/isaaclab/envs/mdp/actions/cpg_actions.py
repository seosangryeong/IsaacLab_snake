from __future__ import annotations
import numpy as np
import torch
from collections.abc import Sequence
import omni.log

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . import actions_cfg


class JointCPGAction(ActionTerm):
    """
    중앙 패턴 생성기(CPG) 기반 액션 Term.

    Raw action (env마다 6차원)
      0 : R_vertical      (홀수 관절 진폭 목표)
      1 : ω_vertical      (내재 주파수)
      2 : θ_vertical      (외부 위상 자극)
      3 : R_horizontal    (짝수 관절 진폭 목표)
      4 : ω_horizontal
      5 : θ_horizontal
    내부 상태
      φ   : 위상  (env × joint)
      r   : 진폭  (env × 2  [vert, horz])
      ṙ  : 진폭 미분
    """

    cfg: actions_cfg.JointCPGActionCfg
    _asset: Articulation

    # ───────────────────────── 초기화 ─────────────────────────
    def __init__(self, cfg: actions_cfg.JointCPGActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        # 관절 ID 및 이름
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)

        # 홀수(수직)·짝수(수평) 관절 인덱스 분류
        self._vertical_ids   = [i for i, n in enumerate(self._joint_names) if int(n[1:]) % 2 == 1]
        self._horizontal_ids = [i for i in range(self._num_joints)              if i not in self._vertical_ids]

        # ── 내부 CPG 상태 버퍼 ──────────────────────────────────
        self._phi    = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._r      = torch.zeros(self.num_envs, 2,               device=self.device)  # [vert, horz]
        self._r_dot  = torch.zeros_like(self._r)

        # 삼중대각 결합행렬 A^T (미학습 상수)  ▶ register_buffer 대신 속성으로
        mu = self.cfg.mu if isinstance(self.cfg.mu, Sequence) else [self.cfg.mu]*self._num_joints
        A  = torch.zeros(self._num_joints, self._num_joints, device=self.device)
        for i in range(self._num_joints):
            if i > 0:                    A[i, i-1] =  mu[i-1]
            A[i, i] = -2*mu[i]
            if i < self._num_joints-1:   A[i, i+1] =  mu[i]
        self._A_T = A.t()           # (n × n) 전치

        # 진폭 ODE 계수 a
        self._a = self.cfg.a

        # Raw / Processed 액션 버퍼
        self._raw_actions      = torch.zeros(self.num_envs, 6,               device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    # ───────────────────── 프로퍼티 ─────────────────────
    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # ──────────────── 액션 처리 파이프라인 ───────────────
    def process_actions(self, actions: torch.Tensor):
        dt = self._env.step_dt
        self._raw_actions.copy_(actions)

        # 1) RL 액션 → R, ω, θ 분리
        R_vert, ω_vert, θ_vert = actions[:, 0], actions[:, 1], actions[:, 2]
        R_horz, ω_horz, θ_horz = actions[:, 3], actions[:, 4], actions[:, 5]

        # (env, joint) 크기로 확장
        ω = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        θ = torch.zeros_like(ω)
        ω[:, self._vertical_ids]    = ω_vert.unsqueeze(1)
        ω[:, self._horizontal_ids]  = ω_horz.unsqueeze(1)
        θ[:, self._vertical_ids]    = θ_vert.unsqueeze(1)
        θ[:, self._horizontal_ids]  = θ_horz.unsqueeze(1)

        # 2) 위상 적분  φ̇ = ω + Aφ + Bθ
        dphi = ω + torch.matmul(self._phi, self._A_T) + self.cfg.B * θ
        self._phi += dt * dphi

        # 3) 진폭 ODE 적분  r̈ = a[(a/4)(R - r) - ṙ]
        a = self._a
        R_target = torch.stack([R_vert, R_horz], dim=1)            # (env, 2)
        r_ddot   = a * ((a/4)*(R_target - self._r) - self._r_dot)
        self._r_dot += dt * r_ddot
        self._r     += dt * self._r_dot

        # 4) 최종 관절 목표 위치 q = r * sinφ
        q = torch.zeros_like(self._phi)
        q[:, self._vertical_ids]   = self._r[:, 0].unsqueeze(1) * torch.sin(self._phi[:, self._vertical_ids])
        q[:, self._horizontal_ids] = self._r[:, 1].unsqueeze(1) * torch.sin(self._phi[:, self._horizontal_ids])

        self._processed_actions.copy_(q)

    # ─────────────── 시뮬레이터에 적용 ────────────────
    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    # ───────────────────── 리셋 ──────────────────────
    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self._phi.zero_(); self._r.zero_(); self._r_dot.zero_(); self._raw_actions.zero_()
        else:
            self._phi[env_ids]     = 0.0
            self._r[env_ids]       = 0.0
            self._r_dot[env_ids]   = 0.0
            self._raw_actions[env_ids] = 0.0
