# from __future__ import annotations

# # from manager_based_env_cfg import ManagerBasedEnvCfg
# import numpy as np
# import torch
# from collections.abc import Sequence
# import omni.log
# from isaaclab.sim.utils import find_matching_prims



# from isaaclab.assets.articulation import Articulation
# from isaaclab.managers.action_manager import ActionTerm
# from . import actions_cfg
# import isaaclab.sim as sim_utils


# from typing import TYPE_CHECKING


# if TYPE_CHECKING:
#     from isaaclab.envs import ManagerBasedEnv

#     from . import actions_cfg

# class JointSineAction(ActionTerm):
#     """사인파 기반 액션 term.
    
#     이 액션 term은 각 환경마다 6개의 파라미터를 사용
    
#       - 수직 조인트 (실제 조인트 이름: j1, j3, j5, ...): amplitude_vertical, frequency_vertical, phase_vertical  
#       - 수평 조인트 (실제 조인트 이름: j2, j4, j6, ...): amplitude_horizontal, frequency_horizontal, phase_horizontal
    
    
#        position = amplitude * sin(2π * frequency * t + (조인트 번호) * phase)
    
#     """
#     cfg: actions_cfg.JointSineActionCfg
#     _asset: Articulation
#     _current_time: float

#     def __init__(self, cfg: actions_cfg.JointSineActionCfg, env: ManagerBasedEnv) -> None:
#         # 기본 ActionTerm 초기화
#         super().__init__(cfg, env)

#         self._joint_ids, self._joint_names = self._asset.find_joints(
#             self.cfg.joint_names, preserve_order=self.cfg.preserve_order
#         )
#         self._num_joints = len(self._joint_ids)
#         omni.log.info(
#             f"Resolved joint names for {self.__class__.__name__}: {self._joint_names} [{self._joint_ids}]"
#         )

#         if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
#             self._joint_ids = list(range(self._num_joints))

#         self._vertical_joint_names = []
#         self._horizontal_joint_names = []
#         for name in self._joint_names:
#             try:
#                 number = int(name[1:])  # 예: j1 -> 1
#             except Exception as e:
#                 omni.log.warn(f"조인트 이름 {name} 실패: {e}")
#                 number = 0
#             if number % 2 == 1:
#                 self._vertical_joint_names.append(name)
#             else:
#                 self._horizontal_joint_names.append(name)
        
#         if isinstance(self._joint_ids, list):
#             self._even_joint_ids = [self._joint_ids[i] for i, name in enumerate(self._joint_names) if int(name[1:]) % 2 == 1]
#             self._odd_joint_ids = [self._joint_ids[i] for i, name in enumerate(self._joint_names) if int(name[1:]) % 2 == 0]
#         else:
#             self._even_joint_ids = list(range(self._num_joints))[::2]
#             self._odd_joint_ids = list(range(self._num_joints))[1::2]

#         self._num_even = len(self._vertical_joint_names)
#         self._num_odd = len(self._horizontal_joint_names)

#         # raw action: 각 환경마다 6차원 
#         # (vertical: amp_vertical, freq_vertical, phase_vertical, horizontal: amp_horizontal, freq_horizontal, phase_horizontal)
#         self._raw_actions = torch.zeros(self.num_envs, 8, device=self.device)
#         # processed action: 각 관절에 대한 값 (num_envs x num_joints)
#         self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

#         # 시뮬레이션 시간 초기화
#         self._current_time = 0.0

#     @property
#     def action_dim(self) -> int:
#         return 8

#     @property
#     def raw_actions(self) -> torch.Tensor:
#         return self._raw_actions

#     @property
#     def processed_actions(self) -> torch.Tensor:
#         return self._processed_actions

#     def update_time(self, dt: float):
#         """내부 시뮬레이션 시간을 dt만큼 업데이트"""
#         self._current_time += dt

#     def process_actions(self, actions: torch.Tensor, additional_joint_values: torch.Tensor = None):
#         # 시뮬레이션 시간 갱신
#         dt = self._env.step_dt
#         self.update_time(dt)

#         # 클리핑
#         clip_ranges = getattr(self.cfg, "clip_ranges", [(-1.0, 1.0)] * 8)
#         actions_clipped = torch.empty_like(actions)
#         for i in range(8):
#             actions_clipped[:, i] = torch.clamp(actions[:, i], min=clip_ranges[i][0], max=clip_ranges[i][1])
#         actions = actions_clipped
#         self._raw_actions[:] = actions

#         # 액션 분해
#         amp_min_v = actions[:, 0]  # 머리 쪽 진폭 (수직)
#         amp_max_v = actions[:, 1]  # 꼬리 쪽 진폭 (수직)
#         freq_v    = actions[:, 2]
#         phase_v   = actions[:, 3]

#         amp_min_h = actions[:, 4]  # 머리 쪽 진폭 (수평)
#         amp_max_h = actions[:, 5]  # 꼬리 쪽 진폭 (수평)
#         freq_h    = actions[:, 6]
#         phase_h   = actions[:, 7]

#         t = torch.full((self.num_envs, 1), self._current_time, device=self.device)

#         # 조인트 인덱스 0~1 정규화
#         vertical_joint_sorted = sorted(self._vertical_joint_names, key=lambda name: int(name[1:]))
#         vertical_numbers = torch.arange(len(vertical_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)
#         vertical_idx = vertical_numbers / (self._num_even - 1)

#         horizontal_joint_sorted = sorted(self._horizontal_joint_names, key=lambda name: int(name[1:]))
#         horizontal_numbers = torch.arange(len(horizontal_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)
#         horizontal_idx = horizontal_numbers / (self._num_odd - 1)

#         # 진폭 보간
#         amp_v = amp_min_v.unsqueeze(1) + (amp_max_v - amp_min_v).unsqueeze(1) * vertical_idx
#         amp_h = amp_min_h.unsqueeze(1) + (amp_max_h - amp_min_h).unsqueeze(1) * horizontal_idx

#         # 위치 계산
#         vertical_pos = amp_v * torch.sin(2 * np.pi * freq_v.unsqueeze(1) * t + vertical_numbers * phase_v.unsqueeze(1))
#         horizontal_pos = amp_h * torch.sin(2 * np.pi * freq_h.unsqueeze(1) * t + horizontal_numbers * phase_h.unsqueeze(1))

#         # 최종 결과 통합
#         processed = torch.zeros(self.num_envs, self._num_joints, device=self.device)
#         for i, name in enumerate(self._joint_names):
#             if int(name[1:]) % 2 == 1:
#                 idx = vertical_joint_sorted.index(name)
#                 processed[:, i] = vertical_pos[:, idx]
#             else:
#                 idx = horizontal_joint_sorted.index(name)
#                 processed[:, i] = horizontal_pos[:, idx]

#         if additional_joint_values is not None:
#             processed += self.cfg.additional_joint_scale * additional_joint_values

#         self._processed_actions.copy_(processed)




#     def apply_actions(self):
#         """계산된 관절 위치를 관절의 목표 위치로 설정"""
#         self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

#     def reset(self, env_ids: Sequence[int] | None = None) -> None:
#         """환경 리셋 시 raw action 값을 0으로 초기화"""
#         if env_ids is None:
#             self._raw_actions.zero_()
#         else:
#             self._raw_actions[env_ids] = 0.0

######################################################################


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
    """수평(짝수) 조인트만 사인파로 제어하는 액션 term.
    
    각 환경마다 4개의 파라미터 사용:
      - amplitude_min, amplitude_max, frequency, phase (모두 수평 조인트용)
      - position = amplitude * sin(2π * frequency * t + (조인트 번호) * phase)
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

        # 수평(짝수) 조인트만 추출
        self._horizontal_joint_names = [name for name in self._joint_names if int(name[1:]) % 2 == 0]
        self._horizontal_joint_ids = [self._joint_ids[i] for i, name in enumerate(self._joint_names) if int(name[1:]) % 2 == 0]
        self._num_horizontal = len(self._horizontal_joint_names)

        # raw action: 4차원 (amp_min, amp_max, freq, phase)
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        # processed action: 각 관절에 대한 값 (num_envs x num_joints)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        self._current_time = 0.0

    @property
    def action_dim(self) -> int:
        return 4

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

        # 클리핑
        clip_ranges = getattr(self.cfg, "clip_ranges", [(-1.0, 1.0)] * 4)
        actions_clipped = torch.empty_like(actions)
        for i in range(4):
            actions_clipped[:, i] = torch.clamp(actions[:, i], min=clip_ranges[i][0], max=clip_ranges[i][1])
        actions = actions_clipped
        self._raw_actions[:] = actions

        amp_min = actions[:, 0]
        amp_max = actions[:, 1]
        freq = actions[:, 2]
        phase = actions[:, 3]

        t = torch.full((self.num_envs, 1), self._current_time, device=self.device)

        # 수평 조인트 인덱스 0~1 정규화
        horizontal_joint_sorted = sorted(self._horizontal_joint_names, key=lambda name: int(name[1:]))
        horizontal_numbers = torch.arange(len(horizontal_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)
        horizontal_idx = horizontal_numbers / (self._num_horizontal - 1) if self._num_horizontal > 1 else torch.zeros_like(horizontal_numbers)

        # 진폭 보간
        amp_h = amp_min.unsqueeze(1) + (amp_max - amp_min).unsqueeze(1) * horizontal_idx

        # 위치 계산
        horizontal_pos = amp_h * torch.sin(2 * np.pi * freq.unsqueeze(1) * t + horizontal_numbers * phase.unsqueeze(1))

        # 전체 조인트에 대해 결과 할당 (짝수만, 나머지는 0)
        processed = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        for i, name in enumerate(self._joint_names):
            if int(name[1:]) % 2 == 0:
                idx = horizontal_joint_sorted.index(name)
                processed[:, i] = horizontal_pos[:, idx]
            else:
                processed[:, i] = 0.0

        if additional_joint_values is not None:
            processed += self.cfg.additional_joint_scale * additional_joint_values

        self._processed_actions.copy_(processed)

    def apply_actions(self):
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0