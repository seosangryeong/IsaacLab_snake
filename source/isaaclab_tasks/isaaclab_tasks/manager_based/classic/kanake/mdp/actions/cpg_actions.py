# your_action_file.py

from __future__ import annotations
import torch
import numpy as np
from collections.abc import Sequence

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from . import actions_cfg

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class JointCPGAction(ActionTerm):
    cfg: actions_cfg.JointCPGActionCfg
    _asset: Articulation

    def __init__(self, cfg: actions_cfg.JointCPGActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        self._joint_ids_horz, _ = self._asset.find_joints(cfg.joint_names_horz)
        self._joint_ids_vert, _ = self._asset.find_joints(cfg.joint_names_vert)
        self._num_horz_joints = len(self._joint_ids_horz)
        self._num_vert_joints = len(self._joint_ids_vert)

        self._joint_ids = self._joint_ids_horz + self._joint_ids_vert
        self.total_joints = self._num_horz_joints + self._num_vert_joints

        self.r_horz = torch.zeros(self.num_envs, self._num_horz_joints, device=self.device)
        self.phi_horz = torch.zeros(self.num_envs, self._num_horz_joints, device=self.device)
        self.r_vert = torch.zeros(self.num_envs, self._num_vert_joints, device=self.device)
        self.phi_vert = torch.zeros(self.num_envs, self._num_vert_joints, device=self.device)

        self.A_horz = self._create_matrix_A(self._num_horz_joints, cfg.mu)
        self.B_horz = self._create_matrix_B(self._num_horz_joints)
        self.A_vert = self._create_matrix_A(self._num_vert_joints, cfg.mu)
        self.B_vert = self._create_matrix_B(self._num_vert_joints)

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.total_joints, device=self.device)

        # 이제 max 값만 스케일링에 사용됩니다.
        self.action_max = torch.tensor(self.cfg.action_max, device=self.device)

    def _create_matrix_A(self, num_joints, mu):
        A = torch.zeros((num_joints, num_joints), device=self.device)
        for i in range(num_joints):
            if i > 0: A[i, i - 1] = mu
            if i < num_joints - 1: A[i, i + 1] = mu
            diag_val = -mu if i == 0 or i == num_joints - 1 else -2 * mu
            A[i, i] = diag_val
        return A

    def _create_matrix_B(self, num_joints):
        B = torch.zeros((num_joints, num_joints - 1), device=self.device)
        if num_joints > 1:
            for i in range(num_joints - 1):
                B[i, i] = 1.0
                B[i + 1, i] = -1.0
        return B

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        
        self.r_horz[env_ids].zero_()
        self.phi_horz[env_ids].zero_()
        self.r_vert[env_ids].zero_()
        self.phi_vert[env_ids].zero_()
        self._raw_actions[env_ids].zero_()

    def process_actions(self, actions: torch.Tensor):
        dt = self._env.step_dt
        self._raw_actions[:] = actions

        # Tanh를 사용하여 모든 액션 요소를 (-1, 1) 사이로 정규화
        tanh_actions = torch.tanh(actions)

        # 정규화된 7차원 벡터를 각 파라미터로 분해
        (
            tanh_R_h, tanh_R_v, tanh_omega,
            tanh_theta_h, tanh_theta_v,
            tanh_delta_h, tanh_delta_v
        ) = torch.chunk(tanh_actions, chunks=7, dim=1)

        # 각 파라미터의 최대값(절대값 기준)을 cfg에서 가져옴
        max_vals = self.action_max
        
        # <<<<<<<<<<<<<<<<<<<<<<<< 요청하신 스케일링 로직 적용 >>>>>>>>>>>>>>>>>>>>>
        # 진폭 (범위: [0, max])
        R_horz      = (tanh_R_h + 1.0) / 2.0 * max_vals[0]
        R_vert      = (tanh_R_v + 1.0) / 2.0 * max_vals[1]
        
        # 주파수, 위상, 오프셋 (범위: [-max, max])
        omega       = tanh_omega * max_vals[2]
        theta_horz  = tanh_theta_h * max_vals[3]
        theta_vert  = tanh_theta_v * max_vals[4]
        delta_horz  = tanh_delta_h * max_vals[5]
        delta_vert  = tanh_delta_v * max_vals[6]
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # --- 수평(Horizontal) CPG 계산 ---
        R_h_b = R_horz.expand(-1, self._num_horz_joints)
        omega_h_b = omega.expand(-1, self._num_horz_joints)
        delta_h_b = delta_horz.expand(-1, self._num_horz_joints)
        theta_h_b = theta_horz.expand(-1, self._num_horz_joints - 1) if self._num_horz_joints > 1 else torch.zeros((actions.shape[0], 0), device=self.device)

        r_dot_h = (self.cfg.a**2 / (4 * (1 + self.cfg.a))) * (R_h_b - self.r_horz)
        phi_dot_h = omega_h_b + self.phi_horz @ self.A_horz.T + theta_h_b @ self.B_horz.T
        
        self.r_horz += r_dot_h * dt
        self.phi_horz += phi_dot_h * dt
        x_horz = self.r_horz * torch.sin(self.phi_horz) + delta_h_b

        # --- 수직(Vertical) CPG 계산 ---
        R_v_b = R_vert.expand(-1, self._num_vert_joints)
        omega_v_b = omega.expand(-1, self._num_vert_joints)
        delta_v_b = delta_vert.expand(-1, self._num_vert_joints)
        theta_v_b = theta_vert.expand(-1, self._num_vert_joints - 1) if self._num_vert_joints > 1 else torch.zeros((actions.shape[0], 0), device=self.device)

        r_dot_v = (self.cfg.a**2 / (4 * (1 + self.cfg.a))) * (R_v_b - self.r_vert)
        phi_dot_v = omega_v_b + self.phi_vert @ self.A_vert.T + theta_v_b @ self.B_vert.T

        self.r_vert += r_dot_v * dt
        self.phi_vert += phi_dot_v * dt
        x_vert = self.r_vert * torch.sin(self.phi_vert) + delta_v_b

        # --- 결과 결합 ---
        combined_psi = torch.cat([x_horz, x_vert], dim=1)
        self._processed_actions.copy_(combined_psi * self.cfg.output_scale)

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )