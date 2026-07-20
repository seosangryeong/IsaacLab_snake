# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    # actions.py 파일에 정의될 설정 클래스를 import 합니다.
    from . import actions_cfg


class ModularJointEffortAction(ActionTerm):
    """
    모듈러 제어를 위한 관절 토크(Effort) 액션 항입니다.

    이 클래스는 다개체 RL 정책에서 출력된 액션 벡터를 받아 각 관절에 대한 토크 명령으로 변환합니다.
    액션 벡터는 (num_envs, num_agents * action_dim_per_agent) 형태로 주어지며,
    각 에이전트가 자신의 관절 축에 대한 토크를 독립적으로 결정합니다.
    """

    cfg: actions_cfg.ModularJointEffortActionCfg
    """이 액션 항의 설정(configuration) 클래스입니다."""
    _asset: Articulation
    """액션이 적용될 에셋(로봇)입니다."""
    _scale: torch.Tensor
    """입력된 액션에 적용될 스케일 텐서입니다. Shape: (1, action_dim)"""

    def __init__(self, cfg: actions_cfg.ModularJointEffortActionCfg, env: ManagerBasedRLEnv):
        # ActionTerm의 생성자를 호출하여 기본 초기화를 수행합니다.
        super().__init__(cfg, env)

        # 에셋(로봇) 가져오기
        self._asset: Articulation = env.scene[self.cfg.asset_name]

        # 설정에서 정의된 관절 이름에 해당하는 ID를 가져옵니다.
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)

        # 중요한 검증: 설정된 관절의 수가 에이전트 설정과 일치하는지 확인합니다.
        if len(self._joint_ids) != self.action_dim:
            raise ValueError(
                f"설정된 관절의 개수({len(self._joint_ids)})가 예상되는 액션 차원({self.action_dim})과 일치하지 않습니다. "
                f"num_agents({self.cfg.num_agents}) * action_dim_per_agent({self.cfg.action_dim_per_agent})를 확인하세요."
            )
        
        # 디버깅을 위한 로그 출력
        omni.log.info(f"'{self.__class__.__name__}'에 대해 인식된 관절: {self._joint_names} (IDs: {self._joint_ids})")

        # 원본 액션과 처리된 액션을 저장할 텐서를 초기화합니다.
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # 스케일 값을 텐서로 변환하여 GPU에 저장합니다. (연산 효율성)
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)


    """
    Properties (속성)
    """

    @property
    def action_dim(self) -> int:
        """이 액션 항의 전체 차원을 반환합니다."""
        return self.cfg.num_agents * self.cfg.action_dim_per_agent

    @property
    def raw_actions(self) -> torch.Tensor:
        """에이전트로부터 받은 원본 액션 값입니다. [-1, 1] 범위입니다."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """스케일이 적용된 후의 액션 값입니다. 실제 토크 값입니다."""
        return self._processed_actions
    
    """
    Operations (동작)
    """

    def process_actions(self, actions: torch.Tensor):
        """액션을 처리하고 내부 버퍼에 저장합니다."""
        # 원본 액션을 저장합니다.
        normalized_actions = torch.tanh(actions)
        # 정규화된 원본 액션을 저장합니다.
        self._raw_actions[:] = normalized_actions
        # 스케일을 적용하여 실제 토크 값으로 변환하고 저장합니다.
        self._processed_actions = self._raw_actions * self._scale


    def apply_actions(self):
        """처리된 액션(토크)을 시뮬레이션의 로봇 관절에 적용합니다."""
        self._asset.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """지정된 환경 ID에 대해 액션 값을 0으로 리셋합니다."""
        # env_ids가 None이면 모든 환경을 리셋합니다.
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0