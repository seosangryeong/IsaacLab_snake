from __future__ import annotations

# from manager_based_env_cfg import ManagerBasedEnvCfg
import numpy as np
import torch
from collections.abc import Sequence
import omni.log
from isaaclab.sim.utils import find_matching_prims



from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from . import actions_cfg
import isaaclab.sim as sim_utils


from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg

class JointSineAction(ActionTerm):
    """사인파 기반 액션 term.
    
    이 액션 term은 각 환경마다 6개의 파라미터를 사용
    
      - 수직 조인트 (실제 조인트 이름: j1, j3, j5, ...): amplitude_vertical, frequency_vertical, phase_vertical  
      - 수평 조인트 (실제 조인트 이름: j2, j4, j6, ...): amplitude_horizontal, frequency_horizontal, phase_horizontal
    
    
       position = amplitude * sin(2π * frequency * t + (조인트 번호) * phase)
    
    """
    cfg: actions_cfg.JointSineActionCfg
    _asset: Articulation
    _current_time: float

    def __init__(self, cfg: actions_cfg.JointSineActionCfg, env: ManagerBasedEnv) -> None:
        # 기본 ActionTerm 초기화
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
                number = int(name[1:])  # 예: j1 -> 1
            except Exception as e:
                omni.log.warn(f"조인트 이름 {name} 실패: {e}")
                number = 0
            if number % 2 == 1:
                self._vertical_joint_names.append(name)
            else:
                self._horizontal_joint_names.append(name)
        
        if isinstance(self._joint_ids, list):
            self._even_joint_ids = [self._joint_ids[i] for i, name in enumerate(self._joint_names) if int(name[1:]) % 2 == 1]
            self._odd_joint_ids = [self._joint_ids[i] for i, name in enumerate(self._joint_names) if int(name[1:]) % 2 == 0]
        else:
            self._even_joint_ids = list(range(self._num_joints))[::2]
            self._odd_joint_ids = list(range(self._num_joints))[1::2]

        self._num_even = len(self._vertical_joint_names)
        self._num_odd = len(self._horizontal_joint_names)

        # raw action: 각 환경마다 6차원 
        # (vertical: amp_vertical, freq_vertical, phase_vertical, horizontal: amp_horizontal, freq_horizontal, phase_horizontal)
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        # processed action: 각 관절에 대한 값 (num_envs x num_joints)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        # 시뮬레이션 시간 초기화
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

    def update_time(self, dt: float):
        """내부 시뮬레이션 시간을 dt만큼 업데이트"""
        self._current_time += dt

    def process_actions(self, actions: torch.Tensor, additional_joint_values: torch.Tensor = None):
        # 환경에서 dt 가져오기
        dt = self._env.step_dt  
        self.update_time(dt)

        # 기존 actions 클립
        clip_ranges = getattr(self.cfg, "clip_ranges", [(-1.0, 1.0)] * 6)
        actions_clipped = torch.empty_like(actions)
        for i in range(6):
            actions_clipped[:, i] = torch.clamp(actions[:, i], min=clip_ranges[i][0], max=clip_ranges[i][1])
        actions = actions_clipped
        self._raw_actions[:] = actions

        # 각 환경별 파라미터 분리 (shape: [num_envs])
        amplitude_vertical = actions[:, 0]
        frequency_vertical = actions[:, 1]
        phase_vertical = actions[:, 2]
        amplitude_horizontal = actions[:, 3]
        frequency_horizontal = actions[:, 4]
        phase_horizontal = actions[:, 5]

        # 현재 시간을 (num_envs, 1) 텐서로 생성
        t = torch.full((self.num_envs, 1), self._current_time, device=self.device)

        # 실제 조인트 이름에서 숫자 추출: 수직, 수평 조인트 각각 정렬하여 추출
        vertical_joint_sorted = sorted(self._vertical_joint_names, key=lambda name: int(name[1:]))
        vertical_numbers = torch.arange(len(vertical_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)
        horizontal_joint_sorted = sorted(self._horizontal_joint_names, key=lambda name: int(name[1:]))
        horizontal_numbers = torch.arange(len(horizontal_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)

        # 수직 조인트 위치 계산
        vertical_pos = amplitude_vertical.unsqueeze(1) * torch.sin(
            2 * np.pi * frequency_vertical.unsqueeze(1) * t + vertical_numbers * phase_vertical.unsqueeze(1)
        )
        # 수평 조인트 위치 계산
        horizontal_pos = amplitude_horizontal.unsqueeze(1) * torch.sin(
            2 * np.pi * frequency_horizontal.unsqueeze(1) * t + horizontal_numbers * phase_horizontal.unsqueeze(1)
        )

        # 원래의 조인트 순서에 맞게 결과 할당
        processed = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        for i, name in enumerate(self._joint_names):
            if int(name[1:]) % 2 == 1:
                idx = vertical_joint_sorted.index(name)
                processed[:, i] = vertical_pos[:, idx]
            else:
                idx = horizontal_joint_sorted.index(name)
                processed[:, i] = horizontal_pos[:, idx]

        # 추가적인 조인트 값을 더함
        if additional_joint_values is not None:
            # 추가적인 조인트 값에 스케일 적용
            processed += self.cfg.additional_joint_scale * additional_joint_values

        # 최종 결과를 _processed_actions에 복사
        self._processed_actions.copy_(processed)



    def apply_actions(self):
        """계산된 관절 위치를 관절의 목표 위치로 설정"""
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """환경 리셋 시 raw action 값을 0으로 초기화"""
        if env_ids is None:
            self._raw_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0