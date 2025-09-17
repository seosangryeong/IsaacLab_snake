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
        dt = self._env.step_dt  
        self.update_time(dt)

        self._raw_actions[:] = actions

        # tanh로 [-0.3, 0.3] 범위 적용
        # vertical
        amp_v = 1.0 + 0.3 * torch.tanh(actions[:, 0])      # [0.7, 1.3]
        freq_v = 1.0 + 0.3 * torch.tanh(actions[:, 1])     # [0.7, 1.3]
        phase_v = 2.0 + 0.3 * torch.tanh(actions[:, 2])    # [1.7, 2.3]
        # horizontal
        amp_h = 2.0 + 0.4 * torch.tanh(actions[:, 3])      # [2.2, 2.8]
        freq_h = 0.5 + 0.4 * torch.tanh(actions[:, 4])     # [0.2, 0.8]
        phase_h = 0.8 + 0.4 * torch.tanh(actions[:, 5])    # [0.7, 1.3]

        amplitude_vertical = amp_v
        frequency_vertical = freq_v
        phase_vertical = phase_v

        amplitude_horizontal = amp_h
        frequency_horizontal = freq_h
        phase_horizontal = phase_h

        t = torch.full((self.num_envs, 1), self._current_time, device=self.device)

        vertical_joint_sorted = sorted(self._vertical_joint_names, key=lambda name: int(name[1:]))
        vertical_numbers = torch.arange(len(vertical_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)
        horizontal_joint_sorted = sorted(self._horizontal_joint_names, key=lambda name: int(name[1:]))
        horizontal_numbers = torch.arange(len(horizontal_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)

        vertical_pos = amplitude_vertical.unsqueeze(1) * torch.sin(
            2 * np.pi * frequency_vertical.unsqueeze(1) * t + vertical_numbers * phase_vertical.unsqueeze(1)
        )
        horizontal_pos = amplitude_horizontal.unsqueeze(1) * torch.sin(
            2 * np.pi * frequency_horizontal.unsqueeze(1) * t + horizontal_numbers * phase_horizontal.unsqueeze(1)
        )

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

# from __future__ import annotations

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
#         self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
#         # processed action: 각 관절에 대한 값 (num_envs x num_joints)
#         self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

#         # 시뮬레이션 시간 초기화
#         self._current_time = 0.0

#     @property
#     def action_dim(self) -> int:
#         return 6

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
#         clip_ranges = getattr(self.cfg, "clip_ranges", [(-1.0, 1.0)] * 6)
#         actions_clipped = torch.empty_like(actions)
#         for i in range(6):
#             actions_clipped[:, i] = torch.clamp(actions[:, i], min=clip_ranges[i][0], max=clip_ranges[i][1])
#         actions = actions_clipped
#         self._raw_actions[:] = actions

#         # 액션 분해
#         amp_v = actions[:, 0]   # 수직 진폭
#         freq_v = actions[:, 1]  # 수직 주파수
#         phase_v = actions[:, 2] # 수직 위상

#         amp_h = actions[:, 3]   # 수평 진폭
#         freq_h = actions[:, 4]  # 수평 주파수
#         phase_h = actions[:, 5] # 수평 위상

#         t = torch.full((self.num_envs, 1), self._current_time, device=self.device)

#         # 조인트 인덱스 가져오기
#         vertical_joint_sorted = sorted(self._vertical_joint_names, key=lambda name: int(name[1:]))
#         vertical_numbers = torch.arange(len(vertical_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)

#         horizontal_joint_sorted = sorted(self._horizontal_joint_names, key=lambda name: int(name[1:]))
#         horizontal_numbers = torch.arange(len(horizontal_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)

#         # 위치 계산
#         vertical_pos = amp_v.unsqueeze(1) * torch.sin(
#             2 * np.pi * freq_v.unsqueeze(1) * t + vertical_numbers * phase_v.unsqueeze(1)
#         )
#         horizontal_pos = amp_h.unsqueeze(1) * torch.sin(
#             2 * np.pi * freq_h.unsqueeze(1) * t + horizontal_numbers * phase_h.unsqueeze(1)
#         )

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

# ######################################################################

# from __future__ import annotations

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
#     """사인파 기반 액션 .
    
#     이 액션 term은 각 환경마다 (조인트 개수 + 4)개의 파라미터를 사용
#       - 각 조인트별 진폭 (num_joints)
#       - 수직 조인트: frequency_vertical, phase_vertical
#       - 수평 조인트: frequency_horizontal, phase_horizontal

#       position = amplitude[joint] * sin(2π * frequency * t + (조인트 번호) * phase)
#     """
#     cfg: actions_cfg.JointSineActionCfg
#     _asset: Articulation
#     _current_time: float

#     def __init__(self, cfg: actions_cfg.JointSineActionCfg, env: ManagerBasedEnv) -> None:
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
        
#         self._num_vertical = len(self._vertical_joint_names)
#         self._num_horizontal = len(self._horizontal_joint_names)

#         # raw action: 각 환경마다 (num_joints + 4)차원
#         # [amplitudes..., freq_vertical, phase_vertical, freq_horizontal, phase_horizontal]
#         self._raw_actions = torch.zeros(self.num_envs, self._num_joints + 4, device=self.device)
#         # processed action: 각 관절에 대한 값 (num_envs x num_joints)
#         self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

#         self._current_time = 0.0

#     @property
#     def action_dim(self) -> int:
#         return self._num_joints + 4

#     @property
#     def raw_actions(self) -> torch.Tensor:
#         return self._raw_actions

#     @property
#     def processed_actions(self) -> torch.Tensor:
#         return self._processed_actions

#     def update_time(self, dt: float):
#         self._current_time += dt

#     def process_actions(self, actions: torch.Tensor, additional_joint_values: torch.Tensor = None):
#         dt = self._env.step_dt
#         self.update_time(dt)
#         # print("clip before action",actions)
#         actions = torch.pi / 2 * torch.tanh(actions)
#         # actions = torch.pi/2 * ( 8*torch.sigmoid(actions)-1 )
#         # print("tanh_actions",actions)
#         # min_vals, _ = actions.min(dim=1, keepdim=True)
#         # max_vals, _ = actions.max(dim=1, keepdim=True)
#         # actions = (actions - min_vals) / (max_vals - min_vals + 1e-8) * 2.0
#         clip_ranges = getattr(self.cfg, "clip_ranges", [(-1.0, 1.0)] * (self._num_joints + 4))
#         actions_clipped = torch.empty_like(actions)
#         for i in range(self._num_joints + 4):
#             actions_clipped[:, i] = torch.clamp(actions[:, i], min=clip_ranges[i][0], max=clip_ranges[i][1])
#         actions = actions_clipped
#         self._raw_actions[:] = actions
#         # print(f"raw_actions: {self._raw_actions}")

#         amplitudes = actions[:, :self._num_joints]  # (num_envs, num_joints)
#         freq_v = actions[:, self._num_joints]       # (num_envs,)
#         phase_v = actions[:, self._num_joints + 1]
#         freq_h = actions[:, self._num_joints + 2]
#         phase_h = actions[:, self._num_joints + 3]

#         t = torch.full((self.num_envs, 1), self._current_time, device=self.device)

#         # 조인트별로 진폭을 나눠서 적용
#         vertical_joint_sorted = sorted(self._vertical_joint_names, key=lambda name: int(name[1:]))
#         vertical_indices = [self._joint_names.index(name) for name in vertical_joint_sorted]
#         vertical_numbers = torch.arange(len(vertical_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)

#         horizontal_joint_sorted = sorted(self._horizontal_joint_names, key=lambda name: int(name[1:]))
#         horizontal_indices = [self._joint_names.index(name) for name in horizontal_joint_sorted]
#         horizontal_numbers = torch.arange(len(horizontal_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)

#         # 각 조인트별 진폭 추출
#         amp_v = amplitudes[:, vertical_indices] if vertical_indices else torch.zeros(self.num_envs, 0, device=self.device)
#         amp_h = amplitudes[:, horizontal_indices] if horizontal_indices else torch.zeros(self.num_envs, 0, device=self.device)

#         # 위치 계산
#         vertical_pos = amp_v * torch.sin(
#             2 * np.pi * freq_v.unsqueeze(1) * t + vertical_numbers * phase_v.unsqueeze(1)
#         ) if vertical_indices else torch.zeros(self.num_envs, 0, device=self.device)
#         horizontal_pos = amp_h * torch.sin(
#             2 * np.pi * freq_h.unsqueeze(1) * t + horizontal_numbers * phase_h.unsqueeze(1)
#         ) if horizontal_indices else torch.zeros(self.num_envs, 0, device=self.device)

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
#         # print(f"processed_actions: {self._processed_actions}")

#     def apply_actions(self):
#         self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

#     def reset(self, env_ids: Sequence[int] | None = None) -> None:
#         if env_ids is None:
#             self._raw_actions.zero_()
#         else:
#             self._raw_actions[env_ids] = 0.0
