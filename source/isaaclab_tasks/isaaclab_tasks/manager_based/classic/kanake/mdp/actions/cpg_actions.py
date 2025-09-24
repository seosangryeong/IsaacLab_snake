from __future__ import annotations

import torch
import numpy as np
from collections.abc import Sequence
import omni.log
from isaaclab.sim.utils import find_matching_prims

from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from . import actions_cfg

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class JointCPGAction(ActionTerm):
    """
    Matsuoka CPG 기반 액션 term.
    - 각 관절마다 하나의 oscillator (extensor–flexor 쌍)
    - RL 액션 u: 각 oscillator에 대한 토닉 입력 (num_joints)
    """
    cfg: actions_cfg.JointCPGActionCfg
    _asset: Articulation
    _current_time: float

    def __init__(self, cfg: actions_cfg.JointCPGActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        # 1) 관절 해상
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)

        # 2) coupling_matrix 설정
        # if not cfg.coupling_matrix:
        #     # num_joints × num_joints zero matrix
        #     W = torch.zeros((self._num_joints, self._num_joints), device=self.device)
        #     for i in range(self._num_joints - 1):
        #         W[i, i+1] = W[i+1, i] = cfg.inhibition
        #     self.W = W
        # else:
        #     self.W = torch.tensor(cfg.coupling_matrix, dtype=torch.float32, device=self.device)


        self.W = torch.zeros(
            (self._num_joints, self._num_joints),
            dtype=torch.float32,
            device=self.device
        )

        # 3) CPG 파라미터 읽기
        self.a       = cfg.inhibition    # mutual inhibition 강도
        self.b       = cfg.self_inhib    # 자기억제 계수
        self.c       = cfg.tone_bias     # free-response bias
        self.tau_r   = cfg.tau_r         # 회로 응답 시간상수
        self.tau_a   = cfg.tau_a         # 적응 시간상수
        self.scale   = cfg.output_scale  # 출력 스케일

        # 4) raw / processed action 텐서
        shape = (self.num_envs, self._num_joints)
        self._raw_actions       = torch.zeros(shape, device=self.device)
        self._processed_actions = torch.zeros(shape, device=self.device)

        # 5) Matsuoka 상태 변수 초기화
        self.x_e = torch.zeros(shape, device=self.device)
        self.y_e = torch.zeros(shape, device=self.device)
        self.x_f = torch.zeros(shape, device=self.device)
        self.y_f = torch.zeros(shape, device=self.device)

        # 6) 내부 시간
        self._current_time = 0.0


    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        shape = (self.num_envs, self._num_joints)
        eps = 1e-2
        if env_ids is None:
            self.x_e = torch.randn(shape, device=self.device) * eps
            self.x_f = torch.randn(shape, device=self.device) * eps
            self.y_e.zero_(); self.y_f.zero_()
            self._raw_actions.zero_()
        else:
            # 선택적 env_ids 처리
            noise = torch.randn((len(env_ids), self._num_joints), device=self.device) * eps
            self.x_e[env_ids] = noise
            self.x_f[env_ids] = -noise
            self.y_e[env_ids].zero_(); self.y_f[env_ids].zero_()
            self._raw_actions[env_ids].zero_()

    def apply_actions(self):
        # 계산된 ψ를 관절 목표 위치로 설정
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )

    def process_actions(self, actions: torch.Tensor, additional_joint_values=None):
        """
        actions: (num_envs, num_joints) 형태의 토닉 입력 u_i
        """
        # 1) 시간 업데이트
        dt = self._env.step_dt
        # print("dt:", dt)
        self._current_time += dt
        # print(f"[CPG DEBUG] W.shape = {self.W.shape}")
        # print(f"[CPG DEBUG] W =\n{self.W.cpu().numpy()}")
        # 2) 토닉 입력 클리핑
        u = torch.clamp(actions, self.cfg.u_min, self.cfg.u_max)
        # print("u:", u)
        self._raw_actions[:] = u

        # 3) 활성화값
        ze = torch.relu(self.x_e)
        zf = torch.relu(self.x_f)

        # 4) Matsuoka 미분방정식 (Euler)
        # extensor
        dx_e = (
            - self.x_e
            - self.a * zf
            - self.b * self.y_e
            - torch.matmul(zf, self.W.T)
            + u
            + self.c
        ) / self.tau_r
        dy_e = (ze - self.y_e) / self.tau_a

        # flexor
        dx_f = (
            - self.x_f
            - self.a * ze
            - self.b * self.y_f
            - torch.matmul(ze, self.W.T)
            + u
            + self.c
        ) / self.tau_r

        # print(f"[CPG DEBUG] u[0]={u[0].cpu().numpy()}")
        # print(f"[CPG DEBUG] c={self.c}, a={self.a}, b={self.b}, tau_r={self.tau_r}")
        # print(f"[CPG DEBUG] dx_e[0]={dx_e[0].detach().cpu().numpy()}")
        # print(f"[CPG DEBUG] dx_f[0]={dx_f[0].detach().cpu().numpy()}")
        dy_f = (zf - self.y_f) / self.tau_a

        # 적분
        self.x_e = self.x_e + dt * dx_e
        self.y_e = self.y_e + dt * dy_e
        self.x_f = self.x_f + dt * dx_f
        self.y_f = self.y_f + dt * dy_f
        # print(f"[CPG DEBUG] x_e[0]={self.x_e[0].detach().cpu().numpy()}")

        # 5) 출력 계산: ψ = scale * (z_e − z_f)
        psi = self.scale * (torch.relu(self.x_e) - torch.relu(self.x_f))
        # print("psi:", psi)

        # 6) 추가 Joint 값 더하기 (optional)
        if additional_joint_values is not None:
            psi = psi + self.cfg.additional_joint_scale * additional_joint_values

        # 7) 결과 복사
        self._processed_actions.copy_(psi)
