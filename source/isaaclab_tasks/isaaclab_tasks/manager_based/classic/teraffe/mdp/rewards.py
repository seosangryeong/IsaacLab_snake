# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.sensors import ContactSensor

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
import torch.nn.functional as F
from . import observations as obs
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv



def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def upright_posture_bonus(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward for maintaining an upright posture.
    로봇의 로컬좌표계 z축과 월드좌표계 z축의 내적. -1에서 1 사이(1에 가까울수록 upright)"""
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)
    # print("up_proj", up_proj)
    return (up_proj > threshold).float()

def upright_posture_shaped_penalty(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Shaped upright posture reward:
    - threshold 이하: 선형 페널티 (반대가 될수록 더 큰 음수)
    - threshold 이상: 선형 보상 (수직에 가까울수록 더 큰 양수)
    """
    up_proj = obs.base_up_proj(env, asset_cfg).squeeze(-1)  
    
    reward = torch.where(
        up_proj >= threshold,
        (up_proj - threshold) / (1.0 - threshold),
        (up_proj - threshold) / (threshold + 1.0) * 2.0  #
    )
    return reward

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    # print("=== ACTION RATE L2 CALLED ===")
    # print("Current Actions:", env.action_manager.action)
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def obstacle_distance_penalty(
    env: ManagerBasedRLEnv,
    obstacle_names: list[str],
    distance_threshold: float = 0.4,
    active_count_attr: str = "_teraffe_active_obstacle_count",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize the robot when its root is too close to an active obstacle root in the xy plane."""
    active_count = int(getattr(env, active_count_attr, len(obstacle_names)))
    active_count = max(0, min(active_count, len(obstacle_names)))
    if active_count == 0:
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    robot_xy = asset.data.root_pos_w[:, :2]

    distances = []
    for obstacle_name in obstacle_names[:active_count]:
        obstacle: RigidObject = env.scene[obstacle_name]
        obstacle_xy = obstacle.data.root_pos_w[:, :2]
        distances.append(torch.linalg.norm(robot_xy - obstacle_xy, dim=-1))

    min_distance = torch.min(torch.stack(distances, dim=-1), dim=-1).values
    return (min_distance < distance_threshold).float()


def navigation_target_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "navigation_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward getting close to the sampled navigation target."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    distance = torch.linalg.norm(command_term.world_command_pos[:, :2] - asset.data.root_pos_w[:, :2], dim=-1)
    return 1.0 - torch.tanh(distance / std)


def navigation_progress_velocity(
    env: ManagerBasedRLEnv,
    command_name: str = "navigation_command",
    slow_radius: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward velocity toward the target, fading out near arrival."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)

    target_vec = command_term.world_command_pos[:, :2] - asset.data.root_pos_w[:, :2]
    distance = torch.linalg.norm(target_vec, dim=-1)
    target_dir = F.normalize(target_vec, dim=-1)
    progress_vel = torch.sum(asset.data.root_lin_vel_w[:, :2] * target_dir, dim=-1)
    slow_scale = torch.clamp(distance / slow_radius, min=0.0, max=1.0)
    return torch.clamp(progress_vel, min=-1.0, max=1.0) * slow_scale


def navigation_stop_near_target(
    env: ManagerBasedRLEnv,
    command_name: str = "navigation_command",
    distance_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for keeping planar speed when the robot is close to the navigation target."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)

    distance = torch.linalg.norm(command_term.world_command_pos[:, :2] - asset.data.root_pos_w[:, :2], dim=-1)
    planar_speed = torch.linalg.norm(asset.data.root_lin_vel_w[:, :2], dim=-1)
    near_target = distance < distance_threshold
    return torch.square(planar_speed) * near_target.float()


def navigation_heading_alignment(
    env: ManagerBasedRLEnv,
    command_name: str = "navigation_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward facing the sampled navigation target."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)

    target_vec = command_term.world_command_pos[:, :2] - asset.data.root_pos_w[:, :2]
    target_dir = F.normalize(target_vec, dim=-1)
    forward_w = math_utils.quat_apply(asset.data.root_quat_w, torch.tensor((1.0, 0.0, 0.0), device=env.device).repeat(env.num_envs, 1))
    return torch.sum(forward_w[:, :2] * target_dir, dim=-1)


def forward_velocity_alignment(
    env: ManagerBasedRLEnv,
    min_speed: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward aligning the robot's forward direction with its planar velocity direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    planar_vel_b = asset.data.root_lin_vel_b[:, :2]
    speed = torch.linalg.norm(planar_vel_b, dim=-1)
    alignment = planar_vel_b[:, 0] / torch.clamp(speed, min=1.0e-6)
    return alignment * (speed > min_speed).float()


def lateral_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for side-slip / crab-walking in the robot body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 1])


def backward_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for moving backward in the robot body frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(-asset.data.root_lin_vel_b[:, 0], min=0.0)


def straight_steer_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str = "navigation_command",
    lateral_threshold: float = 0.25,
    heading_threshold: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["j1_steer", "j2_steer", "j3_steer", "j4_steer"]),
) -> torch.Tensor:
    """Penalize steering angle only when the target is almost straight ahead."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_is_straight = (torch.abs(command[:, 1]) < lateral_threshold) & (torch.abs(command[:, 2]) < heading_threshold)
    steer_deviation = torch.sum(
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=-1,
    )
    return steer_deviation * target_is_straight.float()


def navigation_heading_error_abs(
    env: ManagerBasedRLEnv,
    command_name: str = "navigation_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for not facing the sampled navigation target."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    target_vec = command_term.world_command_pos - asset.data.root_pos_w
    target_yaw = torch.atan2(target_vec[:, 1], target_vec[:, 0])
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.abs(math_utils.wrap_to_pi(target_yaw - yaw))


def navigation_arrival_bonus(
    env: ManagerBasedRLEnv,
    distance_threshold: float = 0.35,
    command_name: str = "navigation_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Sparse bonus for reaching the sampled navigation target."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    distance = torch.linalg.norm(command_term.world_command_pos[:, :2] - asset.data.root_pos_w[:, :2], dim=-1)
    return (distance < distance_threshold).float()
