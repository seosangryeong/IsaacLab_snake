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
    """
    계층적 업데이트 주기를 사용하는 사인파 기반 2-CPG 액션 term.
    - CPG 상태는 고주파(decimation)로 계속 업데이트되어 부드러운 움직임을 보장.
    - RL 정책(목표 파라미터)은 저주파(rl_policy_update_period_s)로 업데이트.
    """
    cfg: actions_cfg.JointCPGActionCfg
    _asset: Articulation

    def __init__(self, cfg: actions_cfg.JointCPGActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        # 관절 해상 (수평/수직 분리)
        self._joint_ids_horz, _ = self._asset.find_joints(cfg.joint_names_horz)
        self._joint_ids_vert, _ = self._asset.find_joints(cfg.joint_names_vert)
        self._num_horz_joints = len(self._joint_ids_horz)
        self._num_vert_joints = len(self._joint_ids_vert)

        self._joint_ids = self._joint_ids_horz + self._joint_ids_vert
        self.total_joints = self._num_horz_joints + self._num_vert_joints

        # CPG 내부 상태 변수
        self.r_horz = torch.zeros(self.num_envs, self._num_horz_joints, device=self.device)
        self.phi_horz = torch.zeros(self.num_envs, self._num_horz_joints, device=self.device)
        self.r_vert = torch.zeros(self.num_envs, self._num_vert_joints, device=self.device)
        self.phi_vert = torch.zeros(self.num_envs, self._num_vert_joints, device=self.device)

        # CPG 커플링 행렬
        self.A_horz = self._create_matrix_A(self._num_horz_joints, cfg.mu)
        self.B_horz = self._create_matrix_B(self._num_horz_joints)
        self.A_vert = self._create_matrix_A(self._num_vert_joints, cfg.mu)
        self.B_vert = self._create_matrix_B(self._num_vert_joints)
        
        # 베이스라인 및 스케일 설정
        baseline_params = [
            0.87,       # R_horz
            0.17,       # R_vert
            np.pi,      # omega
            0.9,        # theta_horz
            1.8,        # theta_vert
            0.0,        # delta_horz
            0.0         # delta_vert
        ]
        self.baseline_params = torch.tensor(baseline_params, device=self.device)
        self.action_scale = torch.tensor(self.cfg.action_scale, device=self.device)

        # 계층적 주기를 위한 변수
        self._active_params = self.baseline_params.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._policy_update_timer = torch.zeros(self.num_envs, device=self.device)

        # 기본 액션 텐서
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.total_joints, device=self.device)

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
        self._policy_update_timer[env_ids] = 0.0
        self._active_params[env_ids] = self.baseline_params

    def process_actions(self, actions: torch.Tensor):
        # <<<<<<<<<<<<<<<<<<<<<<<< 프린트문 1: CPG 업데이트 주기 확인 >>>>>>>>>>>>>>>>>>>>>>>>>
        # 이 함수는 CPG 업데이트 주기(고주파)마다 호출됩니다.
        # 출력되는 시간 간격(예: 0.02초)을 통해 CPG 업데이트 주기를 확인할 수 있습니다.
        # print(f"[CPG Update] Sim Time: {self._env.sim.current_time:.4f} s | Policy Timer: {self._policy_update_timer[0]:.4f} s")
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        dt = self._env.step_dt
        self._raw_actions[:] = actions
        
        self._policy_update_timer += dt
        update_env_ids = torch.where(self._policy_update_timer >= self.cfg.rl_policy_update_period_s)[0]
        
        if len(update_env_ids) > 0:
            # <<<<<<<<<<<<<<<<<<<< 프린트문 2: RL 정책 업데이트 확인 >>>>>>>>>>>>>>>>>>>>>
            # # 이 블록은 RL 정책 업데이트 주기(저주파, 예: 2초)마다 한 번씩만 실행됩니다.
            # print(f"===========================================================")
            # print(f"[RL Policy Update] TRIGGERED at Sim Time: {self._env.sim.current_time:.4f} s")
            # print(f"===========================================================")
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            
            new_actions = actions[update_env_ids]
            tanh_actions = torch.tanh(new_actions)
            
            new_final_params = self.baseline_params + tanh_actions * self.action_scale
            self._active_params[update_env_ids] = new_final_params
            
            self._policy_update_timer[update_env_ids] = 0.0

        (
            R_horz, R_vert, omega,
            theta_horz, theta_vert,
            delta_horz, delta_vert
        ) = torch.chunk(self._active_params, chunks=7, dim=1)

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