# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject

import isaaclab.utils.math as math_utils
import math
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
import torch.nn.functional as F
from . import observations as obs
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv


def kanake_upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture.
    로봇의 로컬좌표계 z축과 월드좌표계 z축의 내적. -1에서 1 사이(1에 가까울수록 upright)"""
    up_proj = obs.base_up_proj_kanake(env, asset_cfg).squeeze(-1)
    # print("up_proj", up_proj)
    return (up_proj > threshold).float()

def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture.
    로봇의 로컬좌표계 z축과 월드좌표계 z축의 내적. -1에서 1 사이(1에 가까울수록 upright)"""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)

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

class BaseXAxisDistanceReward(ManagerTermBase):
    """
    Calculate the signed distances of all bodies from the base의 x축 방향 직선 (월드 좌표계) in XY plane.
    Reward is the sum of threshold-clipped distances (the smaller, the better).
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.threshold = cfg.params.get("threshold", 0.1)

    def calculate_line_from_base_x(
        self,
        base_pos: torch.Tensor,
        base_quat: torch.Tensor,
    ) -> torch.Tensor:
        """
        base 위치와 base의 x축 방향을 이용해 직선 방정식(Ax + By + C = 0) 생성 (XY 평면).
        """
        # base 위치 (월드 좌표계)
        x0, y0 = base_pos[:, 0], base_pos[:, 1]
        # base의 x축 방향 벡터 (월드 좌표계)
        local_x = torch.tensor([1.0, 0.0, 0.0], device=base_quat.device).expand(base_quat.shape[0], 3)
        base_x_dir_w = math_utils.quat_apply(base_quat, local_x)[:, :2]  # [envs, 2]
        dx, dy = base_x_dir_w[:, 0], base_x_dir_w[:, 1]

        # 직선의 방향벡터 (dx, dy)와 base 위치 (x0, y0)로 직선 방정식 생성
        # 직선의 일반형: A(x - x0) + B(y - y0) = 0 → Ax + By + C = 0
        # 여기서 A = -dy, B = dx, C = dy*x0 - dx*y0
        A = -dy
        B = dx
        C = dy * x0 - dx * y0

        return torch.stack([A, B, C], dim=-1)  # [envs, 3]

    def calculate_signed_distances(
        self,
        body_positions: torch.Tensor,
        line_coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """
        거리 = Ax + By + C / sqrt(A^2 + B^2)
        """
        A = line_coefficients[:, 0].unsqueeze(1)
        B = line_coefficients[:, 1].unsqueeze(1)
        C = line_coefficients[:, 2].unsqueeze(1)

        x, y = body_positions[..., 0], body_positions[..., 1]
        signed_distances = (A * x + B * y + C) / torch.sqrt(A**2 + B**2 + 1e-8)
        return signed_distances

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        threshold: float = 0.1,
    ) -> torch.Tensor:
        """
        Calculates the reward based on signed distances of bodies from the base x축 직선 (XY 평면).
        """
        asset: Articulation = env.scene[asset_cfg.name]

        # base 위치와 쿼터니언 (월드 좌표계)
        base_pos = asset.data.root_pos_w[:, :3]  # [envs, 3]
        base_quat = asset.data.root_quat_w       # [envs, 4]

        # 모든 body 위치 (XY 평면)
        body_positions = asset.data.body_pos_w[..., :2]  # [envs, num_bodies, 2]

        # base x축 직선 방정식 계산
        line_coefficients = self.calculate_line_from_base_x(base_pos, base_quat)

        # 각 body의 signed 거리 계산
        signed_distances = self.calculate_signed_distances(body_positions, line_coefficients)

        # threshold를 초과하는 거리에 대해서만 페널티 부여
        clipped_distances = torch.clamp(torch.abs(signed_distances) - threshold, min=0.0)
        penalty = clipped_distances.sum(dim=1)

        return penalty
    
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
        sum_of_distances = signed_distances.sum(dim=1)
        
        clipped_sum = torch.clamp(torch.abs(sum_of_distances) - threshold, min=0.0)

        penalty = torch.square(clipped_sum)

        return penalty
    

class BodyOrderReward(ManagerTermBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "kanake_command",  
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        # 커맨드에서 타겟 위치 추출
        # command = env.command_manager.get_command(command_name)
        # target_pos = command[:, :2]  # shape: [envs, 2]
        # print("target_pos:", target_pos)
        command_term = env.command_manager.get_term(command_name)
        target_pos_w = command_term.world_command_pos[:,:2]
        # print("target_pos:", target_pos_w)

        # 바디 순서: head, link1, ..., link15, tail (총 17개)
        order_names = ["head"] + [f"Link{i}" for i in range(1, 16)] + ["tail"]

        # 각 바디의 (x, y) 위치 추출
        body_positions = []
        for name in order_names:
            idx = asset.body_names.index(name)
            pos = asset.data.body_pos_w[:, idx, :2]
            body_positions.append(pos)
        body_positions = torch.stack(body_positions, dim=1)  # [envs, 17, 2]

        # 각 바디와 타겟 사이의 유클리드 거리 계산
        distances = torch.norm(body_positions - target_pos_w.unsqueeze(1), dim=-1)  # [envs, 17]

        # 가까운 순서대로 인덱스 정렬
        sorted_indices = torch.argsort(distances, dim=1)  # [envs, 17]
        # 각 환경별로 가까운 순서의 바디 이름 리스트 생성
        # for env_idx in range(distances.shape[0]):
        #     sorted_names = [order_names[i] for i in sorted_indices[env_idx].tolist()]
        #     print(f"Env {env_idx} order: {sorted_names}")

        # 인접한 바디 쌍마다 올바른 순서인지 확인: d[i] < d[i+1]
        correct_order = distances[:, :-1] < distances[:, 1:]
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
    


def kanake_position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    # print("des_pos_b", des_pos_b)
    # 로봇 위치를 항상 (0,0,기본높이), 회전은 [1,0,0,0]로 가정
    batch = des_pos_b.shape[0]
    root_pos = torch.zeros(batch, 3, device=des_pos_b.device)
    root_pos[:, 2] = asset.data.default_root_state[:, 2]  # 기본 높이
    root_quat = torch.zeros(batch, 4, device=des_pos_b.device)
    root_quat[:, 0] = 1.0  # [1,0,0,0]
    des_pos_w, _ = combine_frame_transforms(root_pos, root_quat, des_pos_b) #->일단 쿼터니안 안쓰고 포지션값만 사용
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def kanake_position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    des_pos_w = env.command_manager.get_command(command_name)[:, :2]
    curr_pos_w = asset.data.root_pos_w[:, :2]  
    
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

# def kanake_position_command_error_base(
#     env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     asset: RigidObject = env.scene[asset_cfg.name]
#     des_pos_w = env.command_manager.get_command(command_name)[:, :2]
#     # print("des_pos_w", des_pos_w)

#     curr_pos_w = asset.data.root_pos_w[:, :2]  
#     # print("curr_pos_w", curr_pos_w)
#     # dis = torch.norm(curr_pos_w - des_pos_w, dim=1)
#     return torch.norm(curr_pos_w - des_pos_w, dim=1)





class kanake_progress_to_command(ManagerTermBase):
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        if not hasattr(env, "episode_length_buf"):
            raise AttributeError("The environment does not have the 'episode_length_buf' attribute.")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene["robot"]
        command_term = env.command_manager.get_term(command_name)
        
        target_pos_w = command_term.world_command_pos[:, :2]
        current_pos_w = asset.data.root_pos_w[:, :2]
        current_distance = torch.norm(target_pos_w - current_pos_w, dim=1)

        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -current_distance
        reward = self.potentials - self.prev_potentials
        reward[env.episode_length_buf == 0] = 0.0

        return reward

    def reset(self, env_ids: torch.Tensor):
        asset: Articulation = self._env.scene["robot"]
        command_term = self._env.command_manager.get_term(self.cfg.params["command_name"])
        
        target_pos_w = command_term.world_command_pos[env_ids, :2]
        current_pos_w = asset.data.root_pos_w[env_ids, :2]
        distance = torch.norm(target_pos_w - current_pos_w, dim=1)

        self.potentials[env_ids] = -distance
        self.prev_potentials[env_ids] = self.potentials[env_ids]


def kanake_position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    
    distance = env.command_manager.get_term(command_name).metrics["error_pos_2d"]

    

    return 1 - torch.tanh(distance / std)


def kanake_position_command_error_base(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    distance = env.command_manager.get_term(command_name).metrics["error_pos_2d"]
    # print("distance", distance)
    # asset: RigidObject = env.scene[asset_cfg.name]
    # command = env.command_manager.get_command(command_name)
    # des_pos_b = command[:, :2]  # (B, 2) - XY만 사용

    # root_pos = asset.data.root_pos_w[:, :2]       # (B, 2) - XY만 사용
    # root_quat = asset.data.root_quat_w            # (B, 4)

    # # 목표 위치를 월드 프레임(XY)으로 변환
    # des_pos_w, _ = combine_frame_transforms(
    #     torch.cat([root_pos, torch.zeros_like(root_pos[:, :1])], dim=1),  # (B, 3)
    #     root_quat,
    #     torch.cat([des_pos_b, torch.zeros_like(des_pos_b[:, :1])], dim=1) # (B, 3)
    # )
    # des_pos_w_xy = des_pos_w[:, :2]

    # # 현재 위치: 루트 위치(XY)
    # curr_pos_w_xy = root_pos

    # distance = torch.norm(curr_pos_w_xy - des_pos_w_xy, dim=1)

    # return 2.0 / torch.square(distance + 0.7)
    return distance

def kanake_position_command_threshold_reward(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, threshold: float = 0.1) -> torch.Tensor:
    """
    명령 좌표에 threshold 이내로 접근하면 보상.
    """
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
    
    # 현재 위치와 목표 위치 사이의 거리 계산
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    
    # 거리가 임계값(threshold) 이내이면 보상 부여
    reward = torch.where(
        distance <= threshold,
        torch.ones_like(distance),          # 임계값 이내면 1
        torch.zeros_like(distance)          # 임계값 초과면 0
    )
    
    return reward

class kanake_progress_command_reward(ManagerTermBase):
    """Reward for making progress towards the commanded target position."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.potentials = torch.zeros(env.num_envs, device=env.device)
        self.prev_potentials = torch.zeros_like(self.potentials)

    def reset(self, env_ids: torch.Tensor):
        asset: Articulation = self._env.scene["robot"]

        # 현재 command 가져오기
        command = self._env.command_manager.get_command(self.cfg.params["command_name"])
        des_pos_b = command[:, :3]

        # 로봇 기준 frame → world frame 변환
        batch = des_pos_b.shape[0]
        root_pos = torch.zeros(batch, 3, device=des_pos_b.device)
        root_pos[:, 2] = asset.data.default_root_state[:, 2]
        root_quat = torch.zeros(batch, 4, device=des_pos_b.device)
        root_quat[:, 0] = 1.0
        des_pos_w, _ = combine_frame_transforms(root_pos, root_quat, des_pos_b)

        # 현재 위치
        curr_pos_w = asset.data.root_pos_w

        to_target_pos = des_pos_w - curr_pos_w
        to_target_pos[:, 2] = 0.0  # 수평면 projection 

        self.potentials[env_ids] = -torch.norm(to_target_pos[env_ids], p=2, dim=-1) / self._env.step_dt
        self.prev_potentials[env_ids] = self.potentials[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]

        command = env.command_manager.get_command(command_name)
        des_pos_b = command[:, :3]

        batch = des_pos_b.shape[0]
        root_pos = torch.zeros(batch, 3, device=des_pos_b.device)
        root_pos[:, 2] = asset.data.default_root_state[:, 2]
        root_quat = torch.zeros(batch, 4, device=des_pos_b.device)
        root_quat[:, 0] = 1.0
        des_pos_w, _ = combine_frame_transforms(root_pos, root_quat, des_pos_b)

        curr_pos_w = asset.data.root_pos_w

        to_target_pos = des_pos_w - curr_pos_w
        to_target_pos[:, 2] = 0.0  # 수평 projection (선택사항)

        self.prev_potentials[:] = self.potentials[:]
        self.potentials[:] = -torch.norm(to_target_pos, p=2, dim=-1) / env.step_dt

        return self.potentials - self.prev_potentials
    

def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path."""
    # 에셋 가져오기
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # heading 값(스칼라)만 추출 
    heading = command[:, 3]  # 모든 환경에 대한 heading 값 (스칼라)
    
    # heading을 z축 회전 쿼터니언으로 변환
    # 쿼터니언 순서: (w, x, y, z) 형식
    cos_yaw = torch.cos(heading * 0.5)
    sin_yaw = torch.sin(heading * 0.5)
    
    # z축 회전에 대한 쿼터니언 (w, x, y, z)
    des_quat_b = torch.zeros((heading.shape[0], 4), device=heading.device)
    des_quat_b[:, 0] = cos_yaw  # w
    des_quat_b[:, 3] = sin_yaw  # z
    
    # 월드 좌표계로 변환
    des_quat_w = quat_mul(asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7], des_quat_b)
    curr_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]
    
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def orientation_command_error_base(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path (root frame based)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    heading = command[:, 3]  # 각 환경의 yaw 값 (스칼라)

    # z축 회전에 대한 쿼터니언 생성 (w, x, y, z)
    cos_yaw = torch.cos(heading * 0.5)
    sin_yaw = torch.sin(heading * 0.5)
    des_quat_b = torch.zeros((heading.shape[0], 4), device=heading.device)
    des_quat_b[:, 0] = cos_yaw  # w
    des_quat_b[:, 3] = sin_yaw  # z

    # 루트 포즈 사용
    root_quat = asset.data.root_quat_w  # (B, 4)
    # print("root_quat", root_quat[0])
    
    # 바디 기준 쿼터니언을 월드 기준으로 변환
    des_quat_w = quat_mul(root_quat, des_quat_b)

    # 현재 루트 쿼터니언
    curr_quat_w = root_quat

    return quat_error_magnitude(curr_quat_w, des_quat_w)


def head_x_direction_alignment_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold_deg: float = 10.0,  # 허용 각도 (degree)
) -> torch.Tensor:
    """
    head의 로컬 x축 방향이 커맨드의 (x, y) 타겟을 바라보면 리워드
    threshold_deg 이내로 정렬되면 1, 아니면 alignment 값 반환
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # 커맨드의 목표 위치 (월드 좌표계)
    target_xy = command[:, :2]  # shape: [num_envs, 2]
    # 현재 head 위치 (월드 좌표계)
    head_pos = asset.data.head_pos_w[:, :2]  # shape: [num_envs, 2]
    # 현재 head 쿼터니언 (월드 좌표계)
    head_quat = asset.data.head_quat_w  # shape: [num_envs, 4]

    # head의 x축 방향 벡터 (월드 좌표계)
    local_x = torch.tensor([1.0, 0.0, 0.0], device=head_quat.device).expand(head_quat.shape[0], 3)
    head_x_world = math_utils.quat_apply(head_quat, local_x)[:, :2]  # shape: [num_envs, 2]

    # head에서 타겟까지의 방향 벡터 (월드 좌표계)
    to_target = target_xy - head_pos
    to_target_norm = torch.norm(to_target, dim=-1, keepdim=True) + 1e-8
    to_target_dir = to_target / to_target_norm  # shape: [num_envs, 2]

    # head x축 방향 벡터 정규화
    head_x_world_norm = torch.norm(head_x_world, dim=-1, keepdim=True) + 1e-8
    head_x_dir = head_x_world / head_x_world_norm

    # print("head_x_world: ", head_x_world)
    # print("to_target_dir: ", to_target_dir)

    # 두 벡터의 코사인 유사도 (alignment)
    alignment = torch.sum(head_x_dir * to_target_dir, dim=-1)
    alignment = torch.clamp(alignment, -1.0, 1.0)

    # print("alignment: ", alignment)
    # threshold_deg 이내면 1, 아니면 alignment 값
    cos_threshold = torch.cos(torch.tensor(threshold_deg * torch.pi / 180.0, device=env.device))
    reward = torch.where(
        alignment >= cos_threshold,
        torch.ones_like(alignment),
        alignment
    )
    return reward


def camera_x_direction_alignment_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold_deg: float = 10.0,
) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    target_pos_w = command[:, :]
    print("target_pos_w", target_pos_w)

    try:
        head_link_idx = robot.body_names.index("head")
    except ValueError:
        raise ValueError("The robot asset does not have a body named 'head'.")

    head_pos_w = robot.data.body_pos_w[:, head_link_idx]
    head_quat_w = robot.data.body_quat_w[:, head_link_idx]

    num_envs = robot.num_instances

    offset_pos_single = torch.tensor([0.048, 0.0, 0.0], device=env.device)
    offset_quat_single = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

    offset_pos = offset_pos_single.repeat(num_envs, 1)
    offset_quat = offset_quat_single.repeat(num_envs, 1)


    camera_pos_w, camera_quat_w = math_utils.combine_frame_transforms(
        head_pos_w, head_quat_w, offset_pos, offset_quat
    )

    local_x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    camera_x_dir_w = math_utils.quat_apply(camera_quat_w, local_x_axis)

    vec_to_target_w = target_pos_w - camera_pos_w
    dir_to_target_w = F.normalize(vec_to_target_w, p=2, dim=-1)

    alignment = torch.sum(camera_x_dir_w * dir_to_target_w, dim=-1)

    angle_rad = torch.acos(alignment.clamp(-1.0, 1.0))
    angle_deg = torch.rad2deg(angle_rad)

    # print("target_pos_w", target_pos_w)
    # print("camera_pos_w", camera_pos_w)
    # print("camera_x_dir_w: ", camera_x_dir_w)
    # print("dir_to_target_w: ", dir_to_target_w)
    # print("alignment: ", alignment)

    reward = torch.where(angle_deg <= threshold_deg, 1.0, alignment)

    return reward

# def camera_x_direction_alignment_reward(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
#     threshold_deg: float = 10.0,
# ) -> torch.Tensor:


#     sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]


#     command = env.command_manager.get_command(command_name)
#     target_pos_w = command[:, :3]

#     camera_pos_w = sensor.data.pos_w
#     camera_quat_w = sensor.data.quat_w_world


#     local_x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
#     camera_x_dir_w = math_utils.quat_apply(camera_quat_w, local_x_axis)

#     vec_to_target_w = target_pos_w - camera_pos_w
#     dir_to_target_w = F.normalize(vec_to_target_w, p=2, dim=-1)

#     alignment = torch.sum(camera_x_dir_w * dir_to_target_w, dim=-1)

#     angle_rad = torch.acos(alignment.clamp(-1.0, 1.0)) 
#     angle_deg = torch.rad2deg(angle_rad)
#     print(env.scene.sensors.keys())
#     print("camera_x_dir_w: ", camera_x_dir_w)
#     print("dir_to_target_w: ", dir_to_target_w)
#     print("alignment: ", alignment)


#     reward = torch.where(angle_deg <= threshold_deg, 1.0, alignment)

#     return reward

def cube_direction_alignment_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold_deg: float = 10.0,
) -> torch.Tensor:
    """
    큐브(헤드)의 x축 방향이 타겟을 향하도록 하는 리워드
    """
    # 로봇 객체 가져오기
    robot: Articulation = env.scene["robot"]
    
    # 커맨드에서 타겟 위치 가져오기
    command = env.command_manager.get_command(command_name)
    target_pos_w = command[:, :3]  # 타겟 위치 (x, y, z)
    
    try:
        cube_idx = robot.body_names.index("cube")
    except ValueError:
        raise ValueError("The robot asset does not have a body named 'cube'.")
    
    cube_pos_w = robot.data.body_pos_w[:, cube_idx]
    cube_quat_w = robot.data.body_quat_w[:, cube_idx]
    
    local_x_axis = torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    cube_x_dir_w = math_utils.quat_apply(cube_quat_w, local_x_axis)
    
    vec_to_target_w = target_pos_w - cube_pos_w
    dir_to_target_w = F.normalize(vec_to_target_w, p=2, dim=-1)
    
    alignment = torch.sum(cube_x_dir_w * dir_to_target_w, dim=-1)
    alignment = torch.clamp(alignment, -1.0, 1.0)  
    
    angle_rad = torch.acos(alignment)
    angle_deg = torch.rad2deg(angle_rad)
    
    reward = torch.where(angle_deg <= threshold_deg, 1.0, alignment)
    
    return reward


def action_rate_l2_clipped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel with safety clipping."""
    # 현재 액션과 이전 액션 추출
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    
    # NaN 또는 Inf 체크 (안정성을 위해)
    if torch.isnan(current_action).any() or torch.isinf(current_action).any() or \
       torch.isnan(prev_action).any() or torch.isinf(prev_action).any():
        return torch.ones(env.num_envs, device=env.device) * 10.0  
    
    # 차이 계산 및 요소별 클리핑
    action_diff = torch.clamp(current_action - prev_action, min=-10.0, max=10.0)
    
    # L2 계산 
    rate_l2 = torch.sum(torch.square(action_diff), dim=1)
    
    # 최종 결과 클리핑 
    return torch.clamp(rate_l2, max=100.0)


class BaseTargetAlignmentReward(ManagerTermBase):
    """
    로봇 베이스의 x축 방향이 목표 지점을 향하도록 하는 리워드
    """
    
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        # 이전 정렬 상태 저장
        self.prev_alignment = torch.zeros(env.num_envs, device=env.device)
        
    def reset(self, env_ids: torch.Tensor):
        self.prev_alignment[env_ids] = -1.0  
    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        perfect_alignment_deg: float = 10.0,
        smooth_factor: float = 2.0,
        improvement_bonus: float = 0.2
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        
        # 타겟 위치와 로봇 위치 (XY평면)
        target_xy = command[:, :2]  # [num_envs, 2]
        base_pos = asset.data.root_pos_w[:, :2]  # [num_envs, 2]
        base_quat = asset.data.root_quat_w  # [num_envs, 4]
        
        # 로봇 베이스 x축 방향 벡터 (월드 좌표계)
        local_x = torch.tensor([1.0, 0.0, 0.0], device=base_quat.device).expand(base_quat.shape[0], 3)
        base_x_world = math_utils.quat_apply(base_quat, local_x)[:, :2]
        base_x_dir = F.normalize(base_x_world, dim=-1)  # 단위 벡터화
        
        # 타겟 방향 벡터 (베이스 → 타겟)
        to_target = target_xy - base_pos
        to_target_dir = F.normalize(to_target, dim=-1)  # 단위 벡터화
        
        # 두 방향 간 각도 정렬 (코사인 유사도)
        alignment = torch.sum(base_x_dir * to_target_dir, dim=-1)
        alignment = torch.clamp(alignment, -1.0, 1.0)
        
        # 각도 계산 (라디안)
        angle_rad = torch.acos(alignment)
        
        # 보상 계산
        perfect_rad = perfect_alignment_deg * torch.pi / 180.0
        reward_base = torch.exp(-(angle_rad / perfect_rad) ** smooth_factor)
        
        # 이전 상태 대비 개선도 측정
        improvement = torch.clamp(alignment - self.prev_alignment, min=0.0)
        improvement_reward = improvement * improvement_bonus
        
        # 최종 보상 = 기본 보상 + 개선 보상
        reward = reward_base + improvement_reward
        
        # 이전 상태 업데이트
        self.prev_alignment[:] = alignment
        
        return reward
    

### HEAD 리워드

# cube높이
def cube_height_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # sigma: float = 0.1,  
) -> torch.Tensor:
    
    asset: Articulation = env.scene[asset_cfg.name]
    # command = env.command_manager.get_command(command_name)
    error_z = env.command_manager.get_term(command_name).metrics["error_z"]

    # cube_z = asset.data.cube_pose_w[:, 2]  # [num_envs]
    # target_height = command[:, 0]

    # print("head_z: ", head_z)
    # error = cube_z - target_height
    # reward = torch.exp(-torch.square(error_z) / (sigma**2))
    # return reward
    return torch.square(error_z)

# head 수직 속도 페널티
def head_vertical_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    
    try:
        head_link_idx = asset.body_names.index("head")
    except ValueError:
        raise ValueError("The robot asset does not have a body named 'head'.")
    
    head_vel_z = asset.data.body_lin_vel_w[:, head_link_idx, 2]

    return torch.square(head_vel_z)


# head 방향이 수직으로 향하도록 (로컬 -Y축이 월드 Z축과 정렬)
def head_orientation_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    
    asset: Articulation = env.scene[asset_cfg.name]

    try:
        head_link_idx = asset.body_names.index("head")
    except ValueError:
        raise ValueError("The robot asset does not have a body named 'head'.")

    head_quat_w = asset.data.body_quat_w[:, head_link_idx]
    

    # head 링크의 하늘 방향을 로컬 -Y축으로 정의
    local_up_axis = torch.tensor([0.0, -1.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    
    world_up_axis = math_utils.quat_apply(head_quat_w, local_up_axis)
    
    # 월드 좌표계의 위쪽 방향 벡터
    world_z_up_vec = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    # print("world_up_axis: ", world_up_axis)
    # print("world_z_up_vec: ", world_z_up_vec)
    return torch.sum(world_up_axis * world_z_up_vec - 1.0, dim=1)

# def camera_orientation_alignment_reward(
#     env: ManagerBasedRLEnv,
#     command_name: str,
#     threshold_deg: float = 5.0,
# ) -> torch.Tensor:

#     robot: Articulation = env.scene["robot"]
#     command = env.command_manager.get_command(command_name)
#     target_yaw_w = command[:, 1]
#     target_pitch_w = command[:, 2]

#     try:
#         head_link_idx = robot.body_names.index("cube")
#     except ValueError:
#         raise ValueError("The robot asset does not have a body named 'head'.")
        
#     head_pos_w = robot.data.body_pos_w[:, head_link_idx]
#     head_quat_w = robot.data.body_quat_w[:, head_link_idx]

#     offset_pos_single = torch.tensor([0.048, 0.0, 0.0], device=env.device)
#     offset_quat_single = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
#     offset_pos = offset_pos_single.expand(env.num_envs, -1)
#     offset_quat = offset_quat_single.expand(env.num_envs, -1)

#     _ , camera_quat_w = math_utils.combine_frame_transforms(
#         head_pos_w, head_quat_w, offset_pos, offset_quat
#     )

#     target_quat_w = math_utils.quat_from_euler_xyz(
#         torch.zeros_like(target_pitch_w), target_pitch_w, target_yaw_w
#     )

#     roll_correction_rad = math.pi / 2.0
#     correction_rolls = torch.full_like(target_pitch_w, roll_correction_rad)
#     zeros = torch.zeros_like(target_pitch_w)
#     frame_correction_quat = math_utils.quat_from_euler_xyz(correction_rolls, zeros, zeros)

#     final_target_quat_w = math_utils.quat_mul(target_quat_w, frame_correction_quat)

#     camera_quat_inv = math_utils.quat_inv(camera_quat_w)
#     diff_quat = math_utils.quat_mul(final_target_quat_w, camera_quat_inv)

#     angle_rad = 2.0 * torch.acos(torch.abs(diff_quat[:, 0]).clamp(-1.0, 1.0))
#     angle_deg = torch.rad2deg(angle_rad)

#     alignment = torch.cos(angle_rad)
#     reward = torch.where(angle_deg <= threshold_deg, 1.0, alignment)

#     return reward
def cube_z_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    큐브의 로컬 z축이 월드 z축과 정렬되도록 하는 리워드
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # cube의 쿼터니언 추출
    cube_quat_w = asset.data.cube_pose_w[:, 3:7]

    # 큐브의 로컬 z축 벡터
    local_z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    world_z_axis = math_utils.quat_apply(cube_quat_w, local_z_axis)

    # 월드 좌표계의 위쪽 방향 벡터
    world_z_up_vec = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)

    # 내적 결과가 1에 가까울수록 정렬됨
    return torch.sum(world_z_axis * world_z_up_vec - 1.0, dim=1)

def camera_orientation_alignment_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold_deg: float = 5.0,
) -> torch.Tensor:
    """큐브의 방향이 목표 방향(roll, yaw, pitch)과 정렬되도록 하는 리워드"""
    
    robot: Articulation = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    
    # 명령에서 roll, yaw, pitch 추출 (인덱스: 1=roll, 2=yaw, 3=pitch)
    target_roll = command[:, 1]
    target_yaw = command[:, 2]
    target_pitch = command[:, 3]
    
    # 오일러 각에서 쿼터니언 생성
    target_quat_w = math_utils.quat_from_euler_xyz(
        target_roll, target_pitch, target_yaw
    )
    
    try:
        cube_idx = robot.body_names.index("cube")
    except ValueError:
        raise ValueError("The robot asset does not have a body named 'cube'.")
        
    # 큐브의 현재 쿼터니언
    cube_quat_w = robot.data.body_quat_w[:, cube_idx]
    
    # 두 쿼터니언 사이의 차이 계산
    cube_quat_inv = math_utils.quat_inv(cube_quat_w)
    diff_quat = math_utils.quat_mul(target_quat_w, cube_quat_inv)
    
    # 각도 차이 계산
    angle_rad = 2.0 * torch.acos(torch.abs(diff_quat[:, 0]).clamp(-1.0, 1.0))
    angle_deg = torch.rad2deg(angle_rad)
    
    # 리워드 계산
    alignment = torch.cos(angle_rad)
    reward = torch.where(angle_deg <= threshold_deg, 1.0, alignment)
    
    return reward


def orientation_command_error(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path."""

    asset: Articulation = env.scene[asset_cfg.name]

    # 명령 불러오기 및 shape 확인
    command = env.command_manager.get_command(command_name)
    if command.ndim == 3:
        command = command.squeeze(1)
    if command.shape[-1] != 7:
        raise RuntimeError(f"Expected command shape [N,7], but got {command.shape}")

    des_quat_b = command[:, 3:7]  # shape [N, 4]
    asset_quat = asset.data.root_state_w[:, 3:7]  # shape [N, 4]
    des_quat_w = quat_mul(asset_quat, des_quat_b)  # shape [N, 4]

    curr_quat_w = asset.data.cube_pose_w[:, 3:7]  # shape [N, 4]

    return quat_error_magnitude(curr_quat_w, des_quat_w)


def orientation_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    
    error_roll = env.command_manager.get_term(command_name).metrics["error_roll"]
    error_yaw = env.command_manager.get_term(command_name).metrics["error_yaw"]
    error_pitch = env.command_manager.get_term(command_name).metrics["error_pitch"]

    return error_roll**2 + error_yaw**2 + error_pitch**2

def cube_xy_plane_alignment_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold_deg: float = 10.0,
) -> torch.Tensor:
    """
    커맨드의 xy평면 방향과 큐브의 xy평면 방향이 평행하면 리워드 1, 아니면 alignment 값 반환
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # 커맨드의 xy평면 방향 벡터 (예: x축 방향)
    command_dir = command[:, :2]  # shape: [num_envs, 2]
    command_dir = F.normalize(command_dir, dim=-1)

    # 큐브의 x축 방향 벡터 (월드 좌표계, xy평면)
    cube_idx = asset.body_names.index("cube")
    cube_quat_w = asset.data.body_quat_w[:, cube_idx]
    local_x = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)
    cube_x_dir_w = math_utils.quat_apply(cube_quat_w, local_x)[:, :2]
    cube_x_dir_w = F.normalize(cube_x_dir_w, dim=-1)

    # 두 벡터의 코사인 유사도 (alignment)
    alignment = torch.sum(command_dir * cube_x_dir_w, dim=-1)
    alignment = torch.clamp(alignment, -1.0, 1.0)

    # 각도 계산
    angle_rad = torch.acos(alignment)
    angle_deg = torch.rad2deg(angle_rad)

    # threshold_deg 이내면 1, 아니면 alignment 값
    reward = torch.where(angle_deg <= threshold_deg, 1.0, alignment)
    return reward


def forward_velocity_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]

    # vx, vy, vz
    base_lin_vel_w = asset.data.root_state_w[:, 7:10]

    # base 쿼터니안
    base_quat_wxyz = asset.data.root_link_pose_w[:, 3:7]

    # x방향 정의
    forward_vec_b = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)

    # 로봇의 x방향이 실제 월드 좌표계에서 어디를 향하는지
    forward_vec_w = math_utils.quat_apply(base_quat_wxyz, forward_vec_b)

    # 실제 이동 벡터와 로봇이 바라보는 방향 벡터 내적
    forward_velocity = torch.sum(base_lin_vel_w * forward_vec_w, dim=1)

    return forward_velocity 

def velocity_target_alignment_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    타겟-base 벡터와 base 속도벡터의 일치
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    
    # 커맨드 위치 (x, y)
    target_pos_w = command_term.world_command_pos[:,:2]

    # base위치 (x, y)
    base_pos_w = asset.data.root_state_w[:, 0:2]

    # base 속도(x,y)
    base_lin_vel_w = asset.data.root_state_w[:, 7:9]


    # 방향 벡터 계산
    vec_to_target = target_pos_w - base_pos_w
    # 속도 방향 벡터
    velocity_vec = base_lin_vel_w


    dir_to_target = F.normalize(vec_to_target, p=2, dim=-1)
    dir_of_velocity = F.normalize(velocity_vec, p=2, dim=-1)


    cosine_similarity = torch.sum(dir_to_target * dir_of_velocity, dim=-1)

    return cosine_similarity

def speed_towards_target_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_speed: float = 1.5,
) -> torch.Tensor:

    # 에셋(로봇) 및 커맨드 기간(term) 인스턴스 가져오기
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)

    # 월드 좌표계 기준 타겟 위치와 현재 베이스 위치 (XY 평면)
    target_pos_w = command_term.world_command_pos[:, :2]
    base_pos_w = asset.data.root_pos_w[:, :2]

    # 현재 베이스의 선형 속도 (월드 좌표계, XY 평면)
    base_vel_w = asset.data.root_lin_vel_w[:, :2]

    # 베이스에서 타겟을 향하는 방향 벡터 계산ㅁ
    vec_to_target = target_pos_w - base_pos_w
    # 방향 벡터를 정규화하여 단위 벡터로 만듦
    dir_to_target = F.normalize(vec_to_target, p=2, dim=-1)

    # 속도 벡터를 타겟 방향 단위 벡터에 내적(dot product)하여 속도 성분을 계산
    # 결과: 타겟 방향으로의 속도 크기
    projected_velocity = torch.sum(base_vel_w * dir_to_target, dim=-1)

    # 보상 shaping:
    # 1. 타겟 반대 방향으로 움직이면 (projected_velocity < 0) 보상이 0이 되도록 clamp(min=0.0)
    # 2. max_speed로 나누어 보상을 정규화하고, 최대 보상이 1을 넘지 않도록 clamp(max=1.0)
    reward = torch.clamp(projected_velocity / max_speed, min=0.0, max=1.0)

    return reward

def cube_x_axis_target_alignment_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for aligning the cube's local x-axis with the direction towards the command target.

    This reward is calculated by the cosine similarity between the cube's local x-axis vector
    (in the world frame) and the vector from the cube's position to the commanded target position.
    A reward of +1 means the cube's x-axis is pointing directly at the target, while -1 means
    it's pointing directly away.
    """
    # 에셋(로봇) 인스턴스 및 커맨드 가져오기
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    target_pos_w = command_term.world_command_pos[:, :3]  

    try:
        cube_idx = asset.body_names.index("cube")
    except ValueError:
        raise ValueError(f"The asset '{asset_cfg.name}' does not have a body named 'cube'.")
    
    cube_pos_w = asset.data.body_pos_w[:, cube_idx]
    cube_quat= asset.data.body_quat_w[:, cube_idx]

    local_x_axis_b = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    cube_x_dir_w = math_utils.quat_apply(cube_quat, local_x_axis_b)

    vec_to_target_w = target_pos_w - cube_pos_w
    dir_to_target_w = F.normalize(vec_to_target_w, p=2, dim=-1)
    
    cosine_similarity = torch.sum(cube_x_dir_w * dir_to_target_w, dim=-1)
    # print("cosine_similarity: ", cosine_similarity)

    return cosine_similarity


class HeadTargetDistanceReward(ManagerTermBase):
    """
    Calculate the dynamically updated line equation (Ax + By + C = 0) between 'head' and target position,
    and the signed distances of all other bodies from the line. 
    Reward is the sum of threshold-clipped distances (the smaller, the better).
    """

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.threshold = cfg.params.get("threshold", 0.1)

    def calculate_line(
        self,
        head_pos: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ax + By + C = 0.
        head를 x1, y1
        target을 x2, y2
        """
        x1, y1 = head_pos[:, 0], head_pos[:, 1]
        x2, y2 = target_pos[:, 0], target_pos[:, 1]

        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        return torch.stack([A, B, C], dim=-1)  # [envs, 3]

    def calculate_signed_distances(
        self,
        body_positions: torch.Tensor,
        line_coefficients: torch.Tensor,
    ) -> torch.Tensor:
        """
        거리 = Ax + By + C / (A^2 + B^2)^(1/2)
        """
        A = line_coefficients[:, 0].unsqueeze(1)
        B = line_coefficients[:, 1].unsqueeze(1)
        C = line_coefficients[:, 2].unsqueeze(1)

        x, y = body_positions[..., 0], body_positions[..., 1]
        signed_distances = (A * x + B * y + C) / torch.sqrt(A**2 + B**2 + 1e-8)
        return signed_distances

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        threshold: float = 0.1,
    ) -> torch.Tensor:
        """
        Calculates the reward based on signed distances of bodies from the head-target line.
        """
        asset: Articulation = env.scene[asset_cfg.name]

        # head 위치
        head_pos = asset.data.body_pos_w[:, asset.body_names.index("cube"), :2]  # [envs, 2]

        # 타겟 위치 (월드 좌표계, xy만 사용)
        command_term = env.command_manager.get_term(command_name)
        
        # 커맨드로 주어진 목표 위치 (x, y)
        target_pos= command_term.world_command_pos[:, :2]

        # 모든 body 위치
        body_positions = asset.data.body_pos_w[..., :2]  # [envs, num_bodies, 2]

        # head-target 직선 계산
        line_coefficients = self.calculate_line(head_pos, target_pos)

        # 각 body의 signed 거리 계산
        signed_distances = self.calculate_signed_distances(body_positions, line_coefficients)

        # threshold를 초과하는 거리에 대해서만 페널티 부여
        clipped_distances = torch.clamp(torch.abs(signed_distances) - threshold, min=0.0)
        penalty = clipped_distances.sum(dim=1)
        print(penalty)

        return penalty
    
    
def average_body_velocity_alignment_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    모든 바디(링크)의 평균 속도 벡터(XY 평면)가 타겟 방향과 정렬되도록 하는 리워드.
    코사인 유사도 기반, 1에 가까울수록 정렬.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    target_pos_w = command_term.world_command_pos[:, :2]  # [num_envs, 2]

    # 베이스 위치 (월드 좌표계, XY)
    base_pos_w = asset.data.root_state_w[:, :2]  # [num_envs, 2]

    # 타겟 방향 벡터 (베이스 to 타겟)
    vec_to_target = target_pos_w - base_pos_w
    dir_to_target = torch.nn.functional.normalize(vec_to_target, p=2, dim=-1)

    # 모든 바디의 월드 속도 (num_envs, num_bodies, 3)
    body_velocities = asset.data.body_lin_vel_w  # [num_envs, num_bodies, 3]
    # 평균 속도 (XY만)
    avg_vel_xy = torch.mean(body_velocities[..., :2], dim=1)  # [num_envs, 2]
    avg_vel_dir = torch.nn.functional.normalize(avg_vel_xy, p=2, dim=-1)

    # 코사인 유사도 계산
    reward = torch.sum(avg_vel_dir * dir_to_target, dim=-1)
    return reward
