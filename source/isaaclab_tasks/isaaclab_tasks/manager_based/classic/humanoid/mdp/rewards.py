# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
import torch.nn.functional as F
from . import observations as obs
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture.
    로봇의 로컬좌표계 z축과 월드좌표계 z축의 내적. -1에서 1 사이(1에 가까울수록 upright)"""
    up_proj = obs.base_up_proj_kanake(env, asset_cfg).squeeze(-1)
    # print("up_proj", up_proj)
    return (up_proj > threshold).float()

def upright_posture_shaped(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Shaped upright posture reward:
    - 아래는 선형 증가
    - threshold 이후는 모두 1.0 고정
    """
    up_proj = obs.base_up_proj_kanake(env, asset_cfg).squeeze(-1)  # [-1, 1]
    up_proj_clipped = torch.clip(up_proj, min=0.0)  # [0, 1]로 제한
    reward = torch.where(
        up_proj_clipped > threshold,
        torch.ones_like(up_proj_clipped),
        up_proj_clipped / threshold
    )
    return reward

def upright_posture_penalty(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    up_proj = obs.base_up_proj_kanake(env, asset_cfg).squeeze(-1)
    # print("up_proj", up_proj)
    return up_proj - threshold

def upright_posture_bonus_0(
    env: ManagerBasedRLEnv, threshold: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Reward for maintaining an upright posture.
    로봇의 로컬좌표계 z축과 월드좌표계 z축의 내적. 0에 가까울수록 수직에 가까움.
    """
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    # print("up_proj", up_proj)

    reward = torch.where(
        torch.abs(up_proj) <= threshold,  
        1.0,  
        1.0 - torch.abs(up_proj)  
    )

    return reward

def debug(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Debugging reward to print the root_link_quat_w values.
    Always returns 0 to avoid affecting training.
    """
    # Extract the robot asset
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the root link quaternion in world coordinates
    head_forward = asset.data.head_forward
    projected_gravity_b = asset.data.projected_gravity_b #똑바로 서있으면 z축(3번째)데이터가 -1
    # quat2euler=math_utils.euler_xyz_from_quat(root_quat)
    # Print the quaternion for debugging
    # print("root_link_quat_w:", root_quat)
    # print("projected_gravity_b:", projected_gravity_b)
    # print("quat2euler:", quat2euler)
    print("head_forward", head_forward)
    # Return a dummy reward value (0) to avoid affecting training

    return torch.zeros(env.num_envs, device=env.device)

def heading(
    env: ManagerBasedRLEnv,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    head_forward = asset.data.head_forward
    
    target = torch.tensor(target_pos, device=env.device, dtype=head_forward.dtype)
    
    to_target = target - asset.data.body_pos_w[:,0,:]
    
    forward_norm = F.normalize(head_forward, p=2, dim=-1)
    to_target_norm = F.normalize(to_target, p=2, dim=-1)
    
    heading_reward = torch.sum(forward_norm * to_target_norm, dim=-1)
    
    return heading_reward

def base_up_proj1(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Projection of the base up vector onto the world up vector (for debugging)."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute base up vector
    base_up_vec = -asset.data.projected_gravity_b

    # Print values for debugging
    print("base_up_vec:", base_up_vec)
    # print("projected_gravity_b:", -base_up_vec)

    # Return a dummy reward value (0) to avoid affecting training
    return torch.zeros(env.num_envs, device=env.device)



def move_to_target_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    target_pos: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving to the target heading."""
    heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
    return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


class progress_reward(ManagerTermBase):
    """Reward for making progress towards the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # initialize the base class
        super().__init__(cfg, env)
        # create history buffer
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = self._env.scene["robot"]
        # compute projection of current heading to desired heading vector
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
        # reward terms
        self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute vector to target
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0
        # update history buffer and compute new potential
        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt
        # print(env.step_dt)

        return self.potentials - self.prev_potentials
    


class progress_monotonic_reward(ManagerTermBase):
    """Reward for monotonic progress towards the target (best distance so far)."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.best_distance = torch.full((env.num_envs,), float('inf'), device=env.device)

    def reset(self, env_ids: torch.Tensor):
        asset: Articulation = self._env.scene["robot"]
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
        to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
        to_target_pos[:, 2] = 0.0  # z축 제거
        curr_dist = torch.norm(to_target_pos, p=2, dim=-1)
        self.best_distance[env_ids] = curr_dist

    def __call__(self, env: ManagerBasedRLEnv, target_pos: tuple[float, float, float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        target_pos = torch.tensor(target_pos, device=env.device)
        to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
        to_target_pos[:, 2] = 0.0  
        curr_dist = torch.norm(to_target_pos, p=2, dim=-1)
        reward = torch.clamp(self.best_distance - curr_dist, min=0.0)
        self.best_distance = torch.minimum(self.best_distance, curr_dist)
        return reward
    

class progress_x_distance_reward(ManagerTermBase):
    """Reward for forward progress along the x-axis."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.prev_pos_x = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        asset: Articulation = self._env.scene["robot"]
        self.prev_pos_x[env_ids] = asset.data.head_pos_w[:, 0]  # x축 위치 저장

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        curr_pos_x = asset.data.head_pos_w[:, 0]  # 현재 x 위치
        reward = curr_pos_x - self.prev_pos_x
        self.prev_pos_x[:] = curr_pos_x  # 다음 step을 위해 갱신
        return reward
    
class progress_x_reward(ManagerTermBase):
    """Reward for cumulative forward progress from the initial position."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.init_pos_x = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        asset: Articulation = self._env.scene["robot"]
        self.init_pos_x[env_ids] = asset.data.head_pos_w[env_ids, 0]  # 초기 x 저장

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        curr_pos_x = asset.data.head_pos_w[:, 0]
        reward = curr_pos_x - self.init_pos_x
        # print("init_pos_x", self.init_pos_x)
        # print("curr_pos_x", curr_pos_x)
        # print("reward", reward)
        return reward
    

class progress_y_penalty(ManagerTermBase):
    """Y-axis deviation penalty reward term."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.init_pos_y = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        asset: Articulation = self._env.scene["robot"]
        self.init_pos_y[env_ids] = asset.data.head_pos_w[env_ids, 1]  # 초기 y 위치 저장

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        y_threshold: float = 1.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        # 로봇 자산 가져오기
        asset: Articulation = env.scene[asset_cfg.name]
        curr_pos_y = asset.data.head_pos_w[:, 1]
        
        # y축 이탈 계산 (초기 위치와의 차이)
        y_deviation = torch.abs(curr_pos_y - self.init_pos_y)
        
        # 임계값(threshold)을 초과한 경우에만 페널티 적용
        # 초과한 거리가 클수록 페널티도 커짐
        y_penalty = torch.where(
            y_deviation > y_threshold,
            (y_deviation - y_threshold),  # 음수 보상(페널티)
            torch.zeros_like(y_deviation)  # 임계값 이내면 페널티 없음
        )
        
        return y_penalty  # 음수 값 반환



class joint_limits_penalty_ratio(ManagerTermBase):

    """Penalty for violating joint limits weighted by the gear ratio."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        if "asset_cfg" not in cfg.params:
            cfg.params["asset_cfg"] = SceneEntityCfg("robot")
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(
        self, env: ManagerBasedRLEnv, threshold: float, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # compute the penalty over normalized joints
        joint_pos_scaled = math_utils.scale_transform(
            asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
        )
        # scale the violation amount by the gear ratio
        violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
        violation_amount = violation_amount * self.gear_ratio_scaled

        return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


class power_consumption(ManagerTermBase):
    """Penalty for the power consumed by the actions to the environment.

    This is computed as commanded torque times the joint velocity.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # add default argument
        if "asset_cfg" not in cfg.params:
            cfg.params["asset_cfg"] = SceneEntityCfg("robot")
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # resolve the gear ratio for each joint
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        index_list, _, value_list = string_utils.resolve_matching_names_values(
            cfg.params["gear_ratio"], asset.joint_names
        )
        self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

    def __call__(self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
        # extract the used quantities (to enable type-hinting)
        asset: Articulation = env.scene[asset_cfg.name]
        # return power = torque * velocity (here actions: joint torques)
        return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)


class DistanceReward(ManagerTermBase):
    """
    Calculate the dynamically updated line equation (Ax + By + C = 0) between 'head' and 'tail' bodies and
    the signed distances of all other bodies from the line.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # Initialize the base class
        super().__init__(cfg, env)
        # threshold 파라미터 가져오기 (기본값 0.1)
        self.threshold = cfg.params.get("threshold", 0.1)

    def calculate_line(
        self,
        head_pos: torch.Tensor,
        tail_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ax + By + C = 0.
        head를 x1, y1
        tail을 x2, y2
        """
        x1, y1 = head_pos[:, 0], head_pos[:, 1]
        x2, y2 = tail_pos[:, 0], tail_pos[:, 1]

        A = y2 - y1
        B = x1 - x2
        C = x2*y1 - x1*y2 # Ax1 + By1 + C = 0 -> C = -(Ax1 + By1)

        return torch.stack([A, B, C], dim=-1)  # [envs, 3]

    def calculate_signed_distances(
        self,
        body_positions: torch.Tensor,
        line_coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """
        거리 = Ax + By + C / (A^2 + B^2)^(1/2)
        """
        A = line_coefficients[:, 0].unsqueeze(1)  # [envs, 1]
        B = line_coefficients[:, 1].unsqueeze(1)  # [envs, 1]
        C = line_coefficients[:, 2].unsqueeze(1)  # [envs, 1]

        x, y = body_positions[..., 0], body_positions[..., 1]

        # Calculate signed distances
        signed_distances = (A * x + B * y + C) / torch.sqrt(A**2 + B**2 + 1e-8)  # Avoid division by zero
        return signed_distances

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        threshold: float = 0.1,  # threshold를 파라미터로 받음
    ) -> torch.Tensor:
        """
        Main function that calculates the reward based on signed distances of bodies from the head-tail line.
        """
        asset: Articulation = env.scene[asset_cfg.name]

        # 현재 head와 tail의 위치
        current_head_positions = asset.data.body_pos_w[:, asset.body_names.index("head"), :2]  # Head [x, y]
        current_tail_positions = asset.data.body_pos_w[:, asset.body_names.index("tail"), :2]  # Tail [x, y]

        # 모든 body 위치
        current_body_positions = asset.data.body_pos_w[..., :2]  # All bodies [x, y]

        # head-tail 직선 계산 
        line_coefficients = self.calculate_line(current_head_positions, current_tail_positions)

        # 각 body의 signed 거리 계산
        signed_distances = self.calculate_signed_distances(current_body_positions, line_coefficients)

        # threshold를 초과하는 거리에 대해서만 페널티 부여
        clipped_distances = torch.clamp(torch.abs(signed_distances) - threshold, min=0.0)  # threshold 파라미터 사용
        reward = clipped_distances.sum(dim=1)

        

        return reward
    

class BodyOrderReward(ManagerTermBase):
    
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:

        asset: Articulation = env.scene[asset_cfg.name]
        # 타겟 위치에서 (x, y) 좌표만 사용
        target_pos = torch.tensor(target_pos, device=env.device)[:2]

        # 바디 순서: head, link1, ..., link15, tail (총 17개)
        order_names = ["head"] + [f"Link{i}" for i in range(1, 16)] + ["tail"]

        # 각 바디의 (x, y) 위치를 asset.data.body_pos_w에서 추출 (shape: [envs, 2])
        body_positions = []
        for name in order_names:
            idx = asset.body_names.index(name)
            pos = asset.data.body_pos_w[:, idx, :2]
            body_positions.append(pos)
        # shape: [envs, num_bodies (17), 2]
        body_positions = torch.stack(body_positions, dim=1)

        # 각 바디와 타겟 사이의 유클리드 거리 계산 (shape: [envs, 17])
        # 타겟 위치는 모든 env에 대해 동일하므로 unsqueeze로 브로드캐스트
        distances = torch.norm(body_positions - target_pos.unsqueeze(0), dim=-1)

        # 인접한 바디 쌍마다 올바른 순서인지 확인: d[i] < d[i+1] 이어야 함
        correct_order = distances[:, :-1] < distances[:, 1:]
        # 올바른 쌍의 비율 (0~1): 모든 쌍이 올바르면 1, 하나라도 틀리면 그 비율만큼 보상 감소
        reward = correct_order.to(torch.float32).mean(dim=1)

        return reward



class LineAlignmentReward(ManagerTermBase):
    """
    Reward for aligning the line formed by head and tail with the target direction in 2D (x, y),
    with a flat reward range for alignment within a configurable angle threshold.
    """
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)

    def reset(self, env_ids: torch.Tensor):
        asset: Articulation = self._env.scene["robot"]
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        threshold: float = 5.0,  # Threshold in degrees
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """
        Calculate the reward based on the alignment of the head-tail line with the target direction (2D: x, y),
        with a flat reward range for the specified alignment threshold in degrees.
        """
        # Extract target_pos
        asset: Articulation = env.scene[asset_cfg.name]
        target_pos = torch.tensor(target_pos, device=env.device)

        # Get head and tail positions in 2D (x, y)
        head_position = asset.data.body_pos_w[:, asset.body_names.index("head"), :2]  # Head position [x, y]
        tail_position = asset.data.body_pos_w[:, asset.body_names.index("tail"), :2]  # Tail position [x, y]

        # Calculate the direction of the head-tail line (2D)
        line_direction = head_position - tail_position  # Shape: [envs, 2]
        line_direction_norm = torch.norm(line_direction, dim=-1, keepdim=True) + 1e-8
        line_direction = line_direction / line_direction_norm  # Normalize

        # Target direction in 2D (x, y)
        target_direction = target_pos[:2] - head_position  # Shape: [envs, 2]
        target_direction_norm = torch.norm(target_direction, dim=-1, keepdim=True) + 1e-8
        target_direction = target_direction / target_direction_norm  # Normalize

        # Calculate the cosine similarity between line_direction and target_direction
        alignment = torch.sum(line_direction * target_direction, dim=-1)  # Cosine of the angle
        alignment = torch.clamp(alignment, -1.0, 1.0)  # Ensure valid range

        # Convert alignment_threshold (degrees) to cosine similarity
        cos_threshold = torch.cos(torch.tensor(threshold * 3.14159265 / 180.0, device=env.device))

        # Reward logic
        reward = torch.where(
            alignment >= cos_threshold,  # If alignment is within threshold
            torch.ones_like(alignment),  # Assign maximum reward
            alignment  # Otherwise, reward is proportional to alignment
        )

        return reward
    

class VelocityAlignmentReward(ManagerTermBase):
    """
    Reward for aligning the base linear velocity direction with the target direction in 2D (x, y),
    with a flat reward range for alignment within a configurable angle threshold.
    """
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)

    def reset(self, env_ids: torch.Tensor):
        asset: Articulation = self._env.scene["robot"]
        target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        threshold: float = 30.0,  # Threshold in degrees
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """
        Calculate the reward based on the alignment of the base linear velocity direction with the target direction (2D: x, y),
        with a flat reward range for the specified alignment threshold in degrees.
        """
        # Extract target position
        asset: Articulation = env.scene[asset_cfg.name]
        target_pos = torch.tensor(target_pos, device=env.device)

        # Get current base (root) position in 2D (x, y)
        current_pos = asset.data.root_pos_w[:, :2]  # Shape: [envs, 2]

        # 타겟 방향 계산
        target_direction = target_pos[:2] - current_pos  # Shape: [envs, 2]
        target_direction_norm = torch.norm(target_direction, dim=-1, keepdim=True) + 1e-8
        target_direction = target_direction / target_direction_norm  # Normalize

        # base 속도 계산
        velocity = asset.data.root_lin_vel_b[:, :2]  # Shape: [envs, 2]
        velocity_norm = torch.norm(velocity, dim=-1, keepdim=True) + 1e-8
        velocity_direction = velocity / velocity_norm  # Normalize

        # 타겟 방향과 속도 방향의 코사인 유사도 계산
        alignment = torch.sum(velocity_direction * target_direction, dim=-1)  
        alignment = torch.clamp(alignment, -1.0, 1.0)  

        # Convert alignment threshold (in degrees) to cosine similarity threshold
        cos_threshold = torch.cos(torch.tensor(threshold * 3.14159265 / 180.0, device=env.device))

        # if alignment >= cos_threshold, assign full reward (1), otherwise proportional reward.
        reward = torch.where(
            alignment >= cos_threshold,
            torch.ones_like(alignment),
            alignment
        )

        return reward


class HeadTailDistancePenalty(ManagerTermBase):
    """
    Calculate penalty based on the distance between head and tail.
    The closer they are (below the threshold), the higher the penalty.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        min_distance: float = 0.2,  # 최소 허용 거리
    ) -> torch.Tensor:
        """
        Args:
            env: 환경
            asset_cfg: 로봇 설정
            min_distance: head와 tail 사이의 최소 허용 거리 (미터)
        """
        asset: Articulation = env.scene[asset_cfg.name]

        # head와 tail의 위치
        head_pos = asset.data.body_pos_w[:, asset.body_names.index("head"), :2]  # [num_envs, 2]
        tail_pos = asset.data.body_pos_w[:, asset.body_names.index("tail"), :2]  # [num_envs, 2]

        # head-tail 사이 거리 계산
        distance = torch.norm(head_pos - tail_pos, dim=-1)  # [num_envs]

        # 페널티 계산 (거리가 min_distance보다 작을 때만)
        penalty = torch.where(
            distance < min_distance,
            min_distance - distance,  # 거리가 작을수록 페널티 증가
            torch.zeros_like(distance)  # 충분히 멀면 페널티 없음
        )

        return -penalty  # 페널티는 음수 값으로 반환
    
class LocalWorldAlignmentReward(ManagerTermBase):
    """
    로봇의 로컬 좌표계(베이스 프레임)가 월드 좌표계(아이덴티티 쿼터니언)와 일치할 때 보상
    현재 로봇 베이스의 회전(쿼터니언)과 목표 쿼터니언([1, 0, 0, 0]) 간의 차이를 계산
    오차가 작을수록 보상이 커지도록 지수 함수를 사용
    """
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.alpha = cfg.params.get("alpha", 1.0)  # 민감도 상수

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        # 현재 로봇 베이스의 월드 좌표계 상 회전 (쿼터니언, (w, x, y, z) 형식)
        q_current = asset.data.root_quat_w  # shape: [num_envs, 4]
        
        # 목표 쿼터니언: 월드 좌표계와 동일한 방향 (아이덴티티 쿼터니언)
        q_desired = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand_as(q_current)
        
        # 두 쿼터니언의 내적의 절대값을 계산
        dot = torch.abs(torch.sum(q_current * q_desired, dim=-1))
        dot = torch.clamp(dot, 0.0, 1.0)
        
        # 두 쿼터니언 사이의 각 오차 계산 (라디안 단위)
        angle_error = 2 * torch.acos(dot)
        # print("angle_error", angle_error)
        
        # 오차가 작을수록 높은 보상이 나오도록 지수 함수를 적용
        reward = torch.exp(-self.alpha * angle_error)
        
        return reward
    
class JointSmoothnessReward(ManagerTermBase):
    """
    Reward that penalizes large differences between adjacent joint positions.
    Encourages smooth spatial transitions along the snake robot body.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:

        asset: Articulation = env.scene[asset_cfg.name]
        joint_positions = asset.data.joint_pos  # shape: [num_envs, num_joints]

        # 차이 계산: q[i+1] - q[i] → shape: [num_envs, num_joints - 1]
        diffs = joint_positions[:, 1:] - joint_positions[:, :-1]

        # 제곱합: L2 거리의 제곱
        penalty = torch.sum(diffs**2, dim=1)

        return penalty
    
class BodyOrderReward(ManagerTermBase):
    """
    Reward for ensuring that body segments are generally ordered from head to tail 
    based on distance to target, using Spearman's rank correlation.
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        # 타겟 위치에서 (x, y) 좌표만 사용
        target_pos = torch.tensor(target_pos, device=env.device)[:2]

        # 바디 순서: head, link1, ..., link15, tail (총 17개)
        order_names = ["head"] + [f"Link{i}" for i in range(1, 16)] + ["tail"]
        
        # 이상적인 순위: head가 0, tail이 16
        ideal_ranks = torch.arange(len(order_names), device=env.device).float()

        # 각 바디의 (x, y) 위치를 추출
        body_positions = []
        for name in order_names:
            idx = asset.body_names.index(name)
            pos = asset.data.body_pos_w[:, idx, :2]
            body_positions.append(pos)
        body_positions = torch.stack(body_positions, dim=1)  # [envs, num_bodies, 2]

        # 각 바디와 타겟 사이의 거리 계산
        distances = torch.norm(body_positions - target_pos.unsqueeze(0), dim=-1)  # [envs, num_bodies]

        # 각 환경별로 거리에 따른 순위 계산 (가장 가까운 것이 순위 0)
        ranks = torch.zeros_like(distances)
        for i in range(distances.shape[0]):  # 각 환경별로 처리
            ranks[i] = torch.argsort(torch.argsort(distances[i])).float()

        # 스피어만 순위 상관계수 계산
        n = len(order_names)
        rewards = torch.zeros(env.num_envs, device=env.device)
        
        for i in range(env.num_envs):
            # 순위 차이의 제곱합 계산
            d_squared = torch.sum((ranks[i] - ideal_ranks) ** 2)
            # 스피어만 상관계수 계산: ρ = 1 - (6 * Σd²) / (n(n²-1))
            rho = 1.0 - (6.0 * d_squared) / (n * (n**2 - 1))
            
            # 상관계수를 0~1 범위로 변환 (-1~1 → 0~1)
            rewards[i] = (rho + 1.0) / 2.0

        return rewards
    

class BodyLineDistancePenalty(ManagerTermBase):


    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.threshold = cfg.params.get("threshold", 0.1)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_pos: tuple[float, float, float],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        threshold: float = None,
    ) -> torch.Tensor:
        if threshold is None:
            threshold = self.threshold

        asset: Articulation = env.scene[asset_cfg.name]

        # 0,0에서 타겟까지의 직선 (x1, y1)=(0,0), (x2, y2)=target_pos[:2]
        x1, y1 = 0.0, 0.0
        x2, y2 = float(target_pos[0]), float(target_pos[1])

        # 직선 방정식: Ax + By + C = 0
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        denom = (A**2 + B**2) ** 0.5 + 1e-8  # float


        # 모든 바디의 (x, y) 위치
        body_positions = asset.data.body_pos_w[..., :2]  # [num_envs, num_bodies, 2]
        x = body_positions[..., 0]
        y = body_positions[..., 1]

        # 각 바디의 직선까지의 거리 (부호 무시)
        dist = torch.abs(A * x + B * y + C) / denom  # [num_envs, num_bodies]

        # threshold 이내는 0, 초과만 페널티
        penalty = torch.clamp(dist - threshold, min=0.0)
        reward = (penalty**2).sum(dim=1)

        return reward
    




def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # print("command shape:", command)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    # des_quat_b = command[:, 3:7]
    des_quat_b = command
    des_quat_b = des_quat_b.reshape(-1, 4)
    # print("command shape:", command.shape)
    # print("des_quat_b shape:", des_quat_b.shape)
    # print("body_state_w shape:", asset.data.body_state_w.shape)
    # print("asset_cfg.body_ids:", asset_cfg.body_ids)
    # print("selected body_state shape:", asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7].shape)

    # des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    des_quat_w = quat_mul(asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    
    return quat_error_magnitude(curr_quat_w, des_quat_w)

    # asset: RigidObject = env.scene[asset_cfg.name]
    # command = env.command_manager.get_command(command_name)
    # # obtain the desired and current orientations
    # des_quat_b = command[:, 3:7]
    # des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    # curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    # return quat_error_magnitude(curr_quat_w, des_quat_w)


def kanake_position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    # 로봇 위치를 항상 (0,0,기본높이), 회전은 [1,0,0,0]로 가정
    batch = des_pos_b.shape[0]
    root_pos = torch.zeros(batch, 3, device=des_pos_b.device)
    root_pos[:, 2] = asset.data.default_root_state[:, 2]  # 기본 높이
    root_quat = torch.zeros(batch, 4, device=des_pos_b.device)
    root_quat[:, 0] = 1.0  # [1,0,0,0]
    des_pos_w, _ = combine_frame_transforms(root_pos, root_quat, des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def kanake_position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    batch = des_pos_b.shape[0]
    root_pos = torch.zeros(batch, 3, device=des_pos_b.device)
    root_pos[:, 2] = asset.data.default_root_state[:, 2]
    root_quat = torch.zeros(batch, 4, device=des_pos_b.device)
    root_quat[:, 0] = 1.0
    des_pos_w, _ = combine_frame_transforms(root_pos, root_quat, des_pos_b)
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# from __future__ import annotations

# import torch
# from typing import TYPE_CHECKING

# import isaaclab.utils.math as math_utils
# import isaaclab.utils.string as string_utils
# from isaaclab.assets import Articulation
# from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

# from . import observations as obs

# if TYPE_CHECKING:
#     from isaaclab.envs import ManagerBasedRLEnv


# def upright_posture_bonus(
#     env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Reward for maintaining an upright posture.
#     로봇의 로컬좌표계 z축과 월드좌표계 z축의 내적. -1에서 1 사이(1에 가까울수록 upright)"""
#     up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
#     # print("up_proj", up_proj)
#     return (up_proj > threshold).float()


# def move_to_target_bonus(
#     env: ManagerBasedRLEnv,
#     threshold: float,
#     target_pos: tuple[float, float, float],
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Reward for moving to the target heading."""
#     heading_proj = obs.base_heading_proj(env, target_pos, asset_cfg).squeeze(-1)
#     return torch.where(heading_proj > threshold, 1.0, heading_proj / threshold)


# class progress_reward(ManagerTermBase):
#     """Reward for making progress towards the target."""

#     def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
#         # initialize the base class
#         super().__init__(cfg, env)
#         # create history buffer
#         self.potentials = torch.zeros(env.num_envs, device=env.device)
#         self.prev_potentials = torch.zeros_like(self.potentials)

#     def reset(self, env_ids: torch.Tensor):
#         # extract the used quantities (to enable type-hinting)
#         asset: Articulation = self._env.scene["robot"]
#         # compute projection of current heading to desired heading vector
#         target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)
#         to_target_pos = target_pos - asset.data.root_pos_w[env_ids, :3]
#         # reward terms
#         self.potentials[env_ids] = -torch.norm(to_target_pos, p=2, dim=-1) / self._env.step_dt
#         self.prev_potentials[env_ids] = self.potentials[env_ids]

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         target_pos: tuple[float, float, float],
#         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor:
#         # extract the used quantities (to enable type-hinting)
#         asset: Articulation = env.scene[asset_cfg.name]
#         # compute vector to target
#         target_pos = torch.tensor(target_pos, device=env.device)
#         to_target_pos = target_pos - asset.data.root_pos_w[:, :3]
#         to_target_pos[:, 2] = 0.0
#         # update history buffer and compute new potential
#         self.prev_potentials[:] = self.potentials[:]
#         self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt
#         # print(env.step_dt)

#         return self.potentials - self.prev_potentials


# class joint_limits_penalty_ratio(ManagerTermBase):
#     """Penalty for violating joint limits weighted by the gear ratio."""

#     def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
#         # add default argument
#         if "asset_cfg" not in cfg.params:
#             cfg.params["asset_cfg"] = SceneEntityCfg("robot")
#         # extract the used quantities (to enable type-hinting)
#         asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
#         # resolve the gear ratio for each joint
#         self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
#         index_list, _, value_list = string_utils.resolve_matching_names_values(
#             cfg.params["gear_ratio"], asset.joint_names
#         )
#         self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
#         self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

#     def __call__(
#         self, env: ManagerBasedRLEnv, threshold: float, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg
#     ) -> torch.Tensor:
#         # extract the used quantities (to enable type-hinting)
#         asset: Articulation = env.scene[asset_cfg.name]
#         # compute the penalty over normalized joints
#         joint_pos_scaled = math_utils.scale_transform(
#             asset.data.joint_pos, asset.data.soft_joint_pos_limits[..., 0], asset.data.soft_joint_pos_limits[..., 1]
#         )
#         # scale the violation amount by the gear ratio
#         violation_amount = (torch.abs(joint_pos_scaled) - threshold) / (1 - threshold)
#         violation_amount = violation_amount * self.gear_ratio_scaled

#         return torch.sum((torch.abs(joint_pos_scaled) > threshold) * violation_amount, dim=-1)


# class power_consumption(ManagerTermBase):
#     """Penalty for the power consumed by the actions to the environment.

#     This is computed as commanded torque times the joint velocity.
#     """

#     def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
#         # add default argument
#         if "asset_cfg" not in cfg.params:
#             cfg.params["asset_cfg"] = SceneEntityCfg("robot")
#         # extract the used quantities (to enable type-hinting)
#         asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
#         # resolve the gear ratio for each joint
#         self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
#         index_list, _, value_list = string_utils.resolve_matching_names_values(
#             cfg.params["gear_ratio"], asset.joint_names
#         )
#         self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
#         self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)

#     def __call__(self, env: ManagerBasedRLEnv, gear_ratio: dict[str, float], asset_cfg: SceneEntityCfg) -> torch.Tensor:
#         # extract the used quantities (to enable type-hinting)
#         asset: Articulation = env.scene[asset_cfg.name]
#         # return power = torque * velocity (here actions: joint torques)
#         return torch.sum(torch.abs(env.action_manager.action * asset.data.joint_vel * self.gear_ratio_scaled), dim=-1)


# class DistanceReward(ManagerTermBase):
#     """
#     Calculate the dynamically updated line equation (Ax + By + C = 0) between 'head' and 'tail' bodies and
#     the signed distances of all other bodies from the line.
#     """

#     def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
#         # Initialize the base class
#         super().__init__(cfg, env)
#         # threshold 파라미터 가져오기 (기본값 0.1)
#         self.threshold = cfg.params.get("threshold", 0.1)

#     def calculate_line(
#         self,
#         head_pos: torch.Tensor,
#         tail_pos: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Ax + By + C = 0.
#         head를 x1, y1
#         tail을 x2, y2
#         """
#         x1, y1 = head_pos[:, 0], head_pos[:, 1]
#         x2, y2 = tail_pos[:, 0], tail_pos[:, 1]

#         A = y2 - y1
#         B = x1 - x2
#         C = x2*y1 - x1*y2 # Ax1 + By1 + C = 0 -> C = -(Ax1 + By1)

#         return torch.stack([A, B, C], dim=-1)  # [envs, 3]

#     def calculate_signed_distances(
#         self,
#         body_positions: torch.Tensor,
#         line_coefficients: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         거리 = Ax + By + C / (A^2 + B^2)^(1/2)
#         """
#         A = line_coefficients[:, 0].unsqueeze(1)  # [envs, 1]
#         B = line_coefficients[:, 1].unsqueeze(1)  # [envs, 1]
#         C = line_coefficients[:, 2].unsqueeze(1)  # [envs, 1]

#         x, y = body_positions[..., 0], body_positions[..., 1]

#         # Calculate signed distances
#         signed_distances = (A * x + B * y + C) / torch.sqrt(A**2 + B**2 + 1e-8)  # Avoid division by zero
#         return signed_distances

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#         threshold: float = 0.1,  # threshold를 파라미터로 받음
#     ) -> torch.Tensor:
#         """
#         Main function that calculates the reward based on signed distances of bodies from the head-tail line.
#         """
#         asset: Articulation = env.scene[asset_cfg.name]

#         # 현재 head와 tail의 위치
#         current_head_positions = asset.data.body_pos_w[:, asset.body_names.index("head"), :2]  # Head [x, y]
#         current_tail_positions = asset.data.body_pos_w[:, asset.body_names.index("tail"), :2]  # Tail [x, y]

#         # 모든 body 위치
#         current_body_positions = asset.data.body_pos_w[..., :2]  # All bodies [x, y]

#         # head-tail 직선 계산 
#         line_coefficients = self.calculate_line(current_head_positions, current_tail_positions)

#         # 각 body의 signed 거리 계산
#         signed_distances = self.calculate_signed_distances(current_body_positions, line_coefficients)

#         # threshold를 초과하는 거리에 대해서만 페널티 부여
#         clipped_distances = torch.clamp(torch.abs(signed_distances) - threshold, min=0.0)  # threshold 파라미터 사용
#         reward = clipped_distances.sum(dim=1)

        

#         return reward
    

# class BodyOrderReward(ManagerTermBase):
    
#     def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
#         super().__init__(cfg, env)

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         target_pos: tuple[float, float, float],
#         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor:

#         asset: Articulation = env.scene[asset_cfg.name]
#         # 타겟 위치에서 (x, y) 좌표만 사용
#         target_pos = torch.tensor(target_pos, device=env.device)[:2]

#         # 바디 순서: head, link1, ..., link15, tail (총 17개)
#         order_names = ["head"] + [f"Link{i}" for i in range(1, 16)] + ["tail"]

#         # 각 바디의 (x, y) 위치를 asset.data.body_pos_w에서 추출 (shape: [envs, 2])
#         body_positions = []
#         for name in order_names:
#             idx = asset.body_names.index(name)
#             pos = asset.data.body_pos_w[:, idx, :2]
#             body_positions.append(pos)
#         # shape: [envs, num_bodies (17), 2]
#         body_positions = torch.stack(body_positions, dim=1)

#         # 각 바디와 타겟 사이의 유클리드 거리 계산 (shape: [envs, 17])
#         # 타겟 위치는 모든 env에 대해 동일하므로 unsqueeze로 브로드캐스트
#         distances = torch.norm(body_positions - target_pos.unsqueeze(0), dim=-1)

#         # 인접한 바디 쌍마다 올바른 순서인지 확인: d[i] < d[i+1] 이어야 함
#         correct_order = distances[:, :-1] < distances[:, 1:]
#         # 올바른 쌍의 비율 (0~1): 모든 쌍이 올바르면 1, 하나라도 틀리면 그 비율만큼 보상 감소
#         reward = correct_order.to(torch.float32).mean(dim=1)

#         return reward



# class LineAlignmentReward(ManagerTermBase):
#     """
#     Reward for aligning the line formed by head and tail with the target direction in 2D (x, y),
#     with a flat reward range for alignment within a configurable angle threshold.
#     """
#     def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
#         super().__init__(cfg, env)

#     def reset(self, env_ids: torch.Tensor):
#         asset: Articulation = self._env.scene["robot"]
#         target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         target_pos: tuple[float, float, float],
#         threshold: float = 5.0,  # Threshold in degrees
#         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor:
#         """
#         Calculate the reward based on the alignment of the head-tail line with the target direction (2D: x, y),
#         with a flat reward range for the specified alignment threshold in degrees.
#         """
#         # Extract target_pos
#         asset: Articulation = env.scene[asset_cfg.name]
#         target_pos = torch.tensor(target_pos, device=env.device)

#         # Get head and tail positions in 2D (x, y)
#         head_position = asset.data.body_pos_w[:, asset.body_names.index("head"), :2]  # Head position [x, y]
#         tail_position = asset.data.body_pos_w[:, asset.body_names.index("tail"), :2]  # Tail position [x, y]

#         # Calculate the direction of the head-tail line (2D)
#         line_direction = head_position - tail_position  # Shape: [envs, 2]
#         line_direction_norm = torch.norm(line_direction, dim=-1, keepdim=True) + 1e-8
#         line_direction = line_direction / line_direction_norm  # Normalize

#         # Target direction in 2D (x, y)
#         target_direction = target_pos[:2] - head_position  # Shape: [envs, 2]
#         target_direction_norm = torch.norm(target_direction, dim=-1, keepdim=True) + 1e-8
#         target_direction = target_direction / target_direction_norm  # Normalize

#         # Calculate the cosine similarity between line_direction and target_direction
#         alignment = torch.sum(line_direction * target_direction, dim=-1)  # Cosine of the angle
#         alignment = torch.clamp(alignment, -1.0, 1.0)  # Ensure valid range

#         # Convert alignment_threshold (degrees) to cosine similarity
#         cos_threshold = torch.cos(torch.tensor(threshold * 3.14159265 / 180.0, device=env.device))

#         # Reward logic
#         reward = torch.where(
#             alignment >= cos_threshold,  # If alignment is within threshold
#             torch.ones_like(alignment),  # Assign maximum reward
#             alignment  # Otherwise, reward is proportional to alignment
#         )

#         return reward
    

# class VelocityAlignmentReward(ManagerTermBase):
#     """
#     Reward for aligning the base linear velocity direction with the target direction in 2D (x, y),
#     with a flat reward range for alignment within a configurable angle threshold.
#     """
#     def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
#         super().__init__(cfg, env)

#     def reset(self, env_ids: torch.Tensor):
#         asset: Articulation = self._env.scene["robot"]
#         target_pos = torch.tensor(self.cfg.params["target_pos"], device=self.device)

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         target_pos: tuple[float, float, float],
#         threshold: float = 30.0,  # Threshold in degrees
#         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor:
#         """
#         Calculate the reward based on the alignment of the base linear velocity direction with the target direction (2D: x, y),
#         with a flat reward range for the specified alignment threshold in degrees.
#         """
#         # Extract target position
#         asset: Articulation = env.scene[asset_cfg.name]
#         target_pos = torch.tensor(target_pos, device=env.device)

#         # Get current base (root) position in 2D (x, y)
#         current_pos = asset.data.root_pos_w[:, :2]  # Shape: [envs, 2]

#         # 타겟 방향 계산
#         target_direction = target_pos[:2] - current_pos  # Shape: [envs, 2]
#         target_direction_norm = torch.norm(target_direction, dim=-1, keepdim=True) + 1e-8
#         target_direction = target_direction / target_direction_norm  # Normalize

#         # base 속도 계산
#         velocity = asset.data.root_lin_vel_b[:, :2]  # Shape: [envs, 2]
#         velocity_norm = torch.norm(velocity, dim=-1, keepdim=True) + 1e-8
#         velocity_direction = velocity / velocity_norm  # Normalize

#         # 타겟 방향과 속도 방향의 코사인 유사도 계산
#         alignment = torch.sum(velocity_direction * target_direction, dim=-1)  
#         alignment = torch.clamp(alignment, -1.0, 1.0)  

#         # Convert alignment threshold (in degrees) to cosine similarity threshold
#         cos_threshold = torch.cos(torch.tensor(threshold * 3.14159265 / 180.0, device=env.device))

#         # if alignment >= cos_threshold, assign full reward (1), otherwise proportional reward.
#         reward = torch.where(
#             alignment >= cos_threshold,
#             torch.ones_like(alignment),
#             alignment
#         )

#         return reward


# class HeadTailDistancePenalty(ManagerTermBase):
#     """
#     Calculate penalty based on the distance between head and tail.
#     The closer they are (below the threshold), the higher the penalty.
#     """

#     def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
#         super().__init__(cfg, env)

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#         min_distance: float = 0.2,  # 최소 허용 거리
#     ) -> torch.Tensor:
#         """
#         Args:
#             env: 환경
#             asset_cfg: 로봇 설정
#             min_distance: head와 tail 사이의 최소 허용 거리 (미터)
#         """
#         asset: Articulation = env.scene[asset_cfg.name]

#         # head와 tail의 위치
#         head_pos = asset.data.body_pos_w[:, asset.body_names.index("head"), :2]  # [num_envs, 2]
#         tail_pos = asset.data.body_pos_w[:, asset.body_names.index("tail"), :2]  # [num_envs, 2]

#         # head-tail 사이 거리 계산
#         distance = torch.norm(head_pos - tail_pos, dim=-1)  # [num_envs]

#         # 페널티 계산 (거리가 min_distance보다 작을 때만)
#         penalty = torch.where(
#             distance < min_distance,
#             min_distance - distance,  # 거리가 작을수록 페널티 증가
#             torch.zeros_like(distance)  # 충분히 멀면 페널티 없음
#         )

#         return -penalty  # 페널티는 음수 값으로 반환
    
# class LocalWorldAlignmentReward(ManagerTermBase):
#     """
#     로봇의 로컬 좌표계(베이스 프레임)가 월드 좌표계(아이덴티티 쿼터니언)와 일치할 때 보상
#     현재 로봇 베이스의 회전(쿼터니언)과 목표 쿼터니언([1, 0, 0, 0]) 간의 차이를 계산
#     오차가 작을수록 보상이 커지도록 지수 함수를 사용
#     """
#     def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
#         super().__init__(cfg, env)
#         self.alpha = cfg.params.get("alpha", 1.0)  # 민감도 상수

#     def __call__(
#         self,
#         env: ManagerBasedRLEnv,
#         asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     ) -> torch.Tensor:
#         asset: Articulation = env.scene[asset_cfg.name]
#         # 현재 로봇 베이스의 월드 좌표계 상 회전 (쿼터니언, (w, x, y, z) 형식)
#         q_current = asset.data.root_quat_w  # shape: [num_envs, 4]
        
#         # 목표 쿼터니언: 월드 좌표계와 동일한 방향 (아이덴티티 쿼터니언)
#         q_desired = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).expand_as(q_current)
        
#         # 두 쿼터니언의 내적의 절대값을 계산
#         dot = torch.abs(torch.sum(q_current * q_desired, dim=-1))
#         dot = torch.clamp(dot, 0.0, 1.0)
        
#         # 두 쿼터니언 사이의 각 오차 계산 (라디안 단위)
#         angle_error = 2 * torch.acos(dot)
#         # print("angle_error", angle_error)
        
#         # 오차가 작을수록 높은 보상이 나오도록 지수 함수를 적용
#         reward = torch.exp(-self.alpha * angle_error)
        
#         return reward