# from __future__ import annotations

# import numpy as np
# import torch
# from collections.abc import Sequence
# import omni.log
# import math

# from isaaclab.assets.articulation import Articulation
# from isaaclab.managers.action_manager import ActionTerm
# from . import actions_cfg

# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
#     from isaaclab.envs import ManagerBasedEnv

# class JointSineAction(ActionTerm):
#     """
#     [수정] ROS 2 컨트롤러의 'Gait 5' 파라미터와 계산 방식을 그대로 적용하여
#     주기적인 사인파 움직임을 생성합니다.
#     """
#     cfg: actions_cfg.JointSineActionCfg
#     _asset: Articulation

#     def __init__(self, cfg: actions_cfg.JointSineActionCfg, env: ManagerBasedEnv) -> None:
#         self._action_dim = 6
#         super().__init__(cfg, env)

#         # -- 조인트 정보 확인 (기존과 동일)
#         self._joint_ids, self._joint_names = self._asset.find_joints(
#             self.cfg.joint_names, preserve_order=self.cfg.preserve_order
#         )
#         self._num_joints = len(self._joint_ids)
#         # ... (나머지 조인트 분류 로직은 기존과 동일)
#         self._vertical_joint_names = []
#         self._horizontal_joint_names = []
#         for name in self._joint_names:
#             number = int(name[1:])
#             if number % 2 == 1: self._vertical_joint_names.append(name)
#             else: self._horizontal_joint_names.append(name)
#         self._num_vertical = len(self._vertical_joint_names)
#         self._num_horizontal = len(self._horizontal_joint_names)

#         self.base_AmpH_deg = 50.0
#         self.base_AmpV_deg = 10.0
#         self.base_WaveFqH = 0.5
#         self.base_WaveFqV = 1.0
#         self.base_ShapeFqH = 1.0
#         self.base_ShapeFqV = 2.0

#         # -- 텐서 초기화
#         self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
#         self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
#         self._seq = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

#     @property
#     def action_dim(self) -> int:
#         return self._action_dim

#     @property
#     def raw_actions(self) -> torch.Tensor:
#         return self._raw_actions

#     @property
#     def processed_actions(self) -> torch.Tensor:
#         return self._processed_actions

#     def process_actions(self, actions: torch.Tensor, additional_joint_values: torch.Tensor = None):
#         # raw_actions에 에이전트의 액션 저장
#         self._raw_actions.copy_(actions)

#         # [수정] 1. 에이전트 액션을 스케일링하여 '변화량(delta)' 계산
#         # actions.shape = (num_envs, 6)
#         delta_amph = actions[:, 0] * self.cfg.action_scale_amph
#         delta_ampv = actions[:, 1] * self.cfg.action_scale_ampv
#         delta_wavefqh = actions[:, 2] * self.cfg.action_scale_wavefqh
#         delta_wavefqv = actions[:, 3] * self.cfg.action_scale_wavefqv
#         delta_shapefqh = actions[:, 4] * self.cfg.action_scale_shapefqh
#         delta_shapefqv = actions[:, 5] * self.cfg.action_scale_shapefqv

#         # [수정] 2. 기본 파라미터에 변화량을 더해 '최종 파라미터' 계산
#         # 각 파라미터는 이제 (num_envs,) shape을 가진 텐서가 됨
#         final_AmpH_deg = self.base_AmpH_deg + delta_amph
#         final_AmpV_deg = self.base_AmpV_deg + delta_ampv
#         final_WaveFqH = self.base_WaveFqH + delta_wavefqh
#         final_WaveFqV = self.base_WaveFqV + delta_wavefqv
#         final_ShapeFqH = self.base_ShapeFqH + delta_shapefqh
#         final_ShapeFqV = self.base_ShapeFqV + delta_shapefqv

#         # --- 아래는 이전 코드의 계산 로직을 '텐서 연산'에 맞게 수정한 부분 ---
        
#         self._seq += 1
#         CmdTimeStep = self._env.step_dt * 1000.0

#         # 주기적인 시간 계산 (final_WaveFq가 텐서이므로 unsqueeze로 차원 맞춰줌)
#         period_steps_v = 1000.0 / (CmdTimeStep * final_WaveFqV.unsqueeze(1))
#         time_steps_v = self._seq.unsqueeze(1) % period_steps_v
#         time_v = time_steps_v * (CmdTimeStep / 1000.0)

#         period_steps_h = 1000.0 / (CmdTimeStep * final_WaveFqH.unsqueeze(1))
#         time_steps_h = self._seq.unsqueeze(1) % period_steps_h
#         time_h = time_steps_h * (CmdTimeStep / 1000.0)
        
#         # 위상차 계산
#         phase_v = final_ShapeFqV.unsqueeze(1) * 2.0 * np.pi / (self._num_vertical - 1)
#         phase_h = final_ShapeFqH.unsqueeze(1) * 2.0 * np.pi / (self._num_horizontal - 1)

#         # 조인트 인덱스 및 순서
#         vertical_indices, horizontal_indices, vertical_numbers, horizontal_numbers = self._get_joint_indices_and_numbers()

#         # 진폭(degree -> radian)
#         amp_v_rad = torch.deg2rad(final_AmpV_deg).unsqueeze(1)
#         amp_h_rad = torch.deg2rad(final_AmpH_deg).unsqueeze(1)

#         # 최종 조인트 위치 계산
#         vertical_pos = amp_v_rad * torch.sin(2 * np.pi * final_WaveFqV.unsqueeze(1) * time_v + phase_v * vertical_numbers)
#         horizontal_pos = amp_h_rad * torch.sin(2 * np.pi * final_WaveFqH.unsqueeze(1) * time_h + phase_h * horizontal_numbers)

#         processed = torch.zeros(self.num_envs, self._num_joints, device=self.device)
#         processed[:, vertical_indices] = vertical_pos
#         processed[:, horizontal_indices] = horizontal_pos

#         self._processed_actions.copy_(processed)

#     def apply_actions(self):
#         self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

#     def reset(self, env_ids: Sequence[int] | None = None) -> None:
#         if env_ids is None:
#             self._seq.zero_()
#         else:
#             self._seq[env_ids] = 0.0

#     # (Helper method) 코드를 깔끔하게 하기 위해 분리
#     def _get_joint_indices_and_numbers(self):
#         vertical_joint_sorted = sorted(self._vertical_joint_names, key=lambda name: int(name[1:]))
#         vertical_indices = [self._joint_names.index(name) for name in vertical_joint_sorted]
#         vertical_numbers = torch.arange(len(vertical_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)

#         horizontal_joint_sorted = sorted(self._horizontal_joint_names, key=lambda name: int(name[1:]))
#         horizontal_indices = [self._joint_names.index(name) for name in horizontal_joint_sorted]
#         horizontal_numbers = torch.arange(len(horizontal_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)
#         return vertical_indices, horizontal_indices, vertical_numbers, horizontal_numbers

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
    [
    
    이 액션 term은 각 환경마다 (조인트 개수 + 4)개의 파라미터를 사용
      - 각 조인트별 진폭 (num_joints)
      - 수직 조인트: frequency_vertical, phase_vertical
      - 수평 조인트: frequency_horizontal, phase_horizontal

      position = amplitude[joint] * sin(2π * frequency * t + (조인트 번호) * phase)
      
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

        self._raw_actions = torch.zeros(self.num_envs, self._num_joints + 4, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

        self._current_time = 0.0

    @property
    def action_dim(self) -> int:
        return self._num_joints + 4

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

        # RL 에이전트의 원본 출력을 raw_actions에 저장
        self._raw_actions[:] = actions


        amp_actions = actions[:, :self._num_joints]
        freq_v_action = actions[:, self._num_joints]
        phase_v_action = actions[:, self._num_joints + 1]
        freq_h_action = actions[:, self._num_joints + 2]
        phase_h_action = actions[:, self._num_joints + 3]


        # # 진폭 (Amplitude)
        # amplitudes = 1.0 *(torch.tanh(amp_actions)+ 1.0)

        # # 주파수 (Frequency)
        # freq_v = 1.0 * (torch.tanh(freq_v_action) + 1.0) 
        # freq_h = 1.0 * (torch.tanh(freq_h_action) + 1.0) 

        # # 위상 (Phase)
        # phase_v = np.pi * torch.tanh(phase_v_action) 
        # phase_h = np.pi * torch.tanh(phase_h_action)
        # amplitudes = 0.5 +(torch.tanh(amp_actions)+ 1.0)
        ########
        # # 주파수 (Frequency)
        # freq_v = 0.3 + 0.5 * (torch.tanh(freq_v_action) + 1.0) / 2.0
        # freq_h = 0.3 + 0.5 * (torch.tanh(freq_h_action) + 1.0) / 2.0

        # # 위상 (Phase)
        # phase_v = np.pi/4 * torch.tanh(phase_v_action) 
        # phase_h = np.pi/4 * torch.tanh(phase_h_action)

        ########
        #진폭 : 0.5~1.5
        amplitudes = 0.5 +(torch.tanh(amp_actions)+ 1.0) / 2.0

        # 주파수 : 0.3 ~ 1.1
        freq_v = 0.5 + 0.8 * (torch.tanh(freq_v_action) + 1.0) / 2.0
        freq_h = 0.5 + 0.8 * (torch.tanh(freq_h_action) + 1.0) / 2.0
        # freq_v = 1.1 * (torch.tanh(freq_v_action)) 
        # freq_h = 1.1 * (torch.tanh(freq_h_action)) 

        # 위상 (Phase)
        phase_v = np.pi/4 * torch.tanh(phase_v_action) 
        phase_h = np.pi/4 * torch.tanh(phase_h_action) 
        
        # # 진폭 (Amplitude)
        # amplitudes = 1.0 *torch.tanh(amp_actions)

        # # 주파수 (Frequency)
        # freq_v = (torch.tanh(freq_v_action) + 1.0) * 1.5
        # freq_h = (torch.tanh(freq_h_action) + 1.0) * 1.5

        # # 위상 (Phase)
        # phase_v = 1.5 + torch.tanh(phase_v_action) 
        # phase_h = 1.5 + torch.tanh(phase_h_action)
        
        t = torch.full((self.num_envs, 1), self._current_time, device=self.device)

        # 조인트별로 진폭을 나눠서 적용
        vertical_joint_sorted = sorted(self._vertical_joint_names, key=lambda name: int(name[1:]))
        vertical_indices = [self._joint_names.index(name) for name in vertical_joint_sorted]
        # vertical_numbers = torch.arange(len(vertical_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)
        vertical_numbers = torch.arange(len(vertical_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)


        horizontal_joint_sorted = sorted(self._horizontal_joint_names, key=lambda name: int(name[1:]))
        horizontal_indices = [self._joint_names.index(name) for name in horizontal_joint_sorted]
        # horizontal_numbers = torch.arange(len(horizontal_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)
        horizontal_numbers = torch.arange(len(horizontal_joint_sorted), device=self.device, dtype=torch.float32).unsqueeze(0)

        # 각 조인트별 진폭 추출 (스케일링된 amplitudes 사용)
        amp_v = amplitudes[:, vertical_indices] if vertical_indices else torch.zeros(self.num_envs, 0, device=self.device)
        amp_h = amplitudes[:, horizontal_indices] if horizontal_indices else torch.zeros(self.num_envs, 0, device=self.device)

        # 위치 계산 (스케일링된 freq, phase 등 사용)
        vertical_pos = amp_v * torch.sin(
            2 * np.pi * freq_v.unsqueeze(1) * t + vertical_numbers * phase_v.unsqueeze(1)
        ) if vertical_indices else torch.zeros(self.num_envs, 0, device=self.device)
        
        horizontal_pos = amp_h * torch.sin(
            2 * np.pi * freq_h.unsqueeze(1) * t + horizontal_numbers * phase_h.unsqueeze(1)
        ) if horizontal_indices else torch.zeros(self.num_envs, 0, device=self.device)

        # 최종 결과 통합
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
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
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
