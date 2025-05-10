# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from . import observations as obs

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture.
    로봇의 로컬좌표계 z축과 월드좌표계 z축의 내적. -1에서 1 사이(1에 가까울수록 upright)"""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    # print("up_proj", up_proj)
    return (up_proj > threshold).float()


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
