from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils


def randomize_obstacle_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    obstacle_names: list[str],
    active_count_attr: str | None = None,
    x_range: tuple[float, float] = (-4.0, 4.0),
    y_range: tuple[float, float] = (-4.0, 4.0),
    min_robot_distance: float = 1.5,
    min_obstacle_distance: float = 1.25,
    z: float = 0.5,
    inactive_z: float = 50.0,
    max_attempts: int = 64,
):
    """Randomize kinematic obstacle poses while keeping a clear area around the robot start."""

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    active_count = len(obstacle_names)
    if active_count_attr is not None:
        active_count = int(getattr(env, active_count_attr, 0))
    active_count = max(0, min(active_count, len(obstacle_names)))

    num_envs = len(env_ids)
    device = env.device
    obstacle_xy = torch.empty((max(active_count, 1), num_envs, 2), device=device)

    for obstacle_id in range(active_count):
        xy = torch.empty((num_envs, 2), device=device)
        accepted = torch.zeros(num_envs, dtype=torch.bool, device=device)

        for _ in range(max_attempts):
            samples = torch.empty((num_envs, 2), device=device)
            samples[:, 0].uniform_(x_range[0], x_range[1])
            samples[:, 1].uniform_(y_range[0], y_range[1])

            far_from_robot = torch.linalg.norm(samples, dim=-1) >= min_robot_distance
            if obstacle_id > 0:
                previous_xy = obstacle_xy[:obstacle_id]
                distances = torch.linalg.norm(previous_xy - samples.unsqueeze(0), dim=-1)
                far_from_obstacles = torch.all(distances >= min_obstacle_distance, dim=0)
            else:
                far_from_obstacles = torch.ones(num_envs, dtype=torch.bool, device=device)

            valid = far_from_robot & far_from_obstacles & ~accepted
            xy[valid] = samples[valid]
            accepted |= valid
            if torch.all(accepted):
                break

        if not torch.all(accepted):
            remaining = ~accepted
            fallback = torch.empty((remaining.sum(), 2), device=device)
            fallback[:, 0].uniform_(x_range[0], x_range[1])
            fallback[:, 1].uniform_(y_range[0], y_range[1])
            xy[remaining] = fallback

        obstacle_xy[obstacle_id] = xy

    for obstacle_id, obstacle_name in enumerate(obstacle_names):
        obstacle: RigidObject = env.scene[obstacle_name]
        root_state = obstacle.data.default_root_state[env_ids].clone()

        positions = root_state[:, 0:3]
        if obstacle_id < active_count:
            positions[:, 0:2] = env.scene.env_origins[env_ids, 0:2] + obstacle_xy[obstacle_id]
            positions[:, 2] = env.scene.env_origins[env_ids, 2] + z
        else:
            positions[:, 0:2] = env.scene.env_origins[env_ids, 0:2]
            positions[:, 2] = env.scene.env_origins[env_ids, 2] + inactive_z

        orientations = root_state[:, 3:7]
        velocities = torch.zeros_like(root_state[:, 7:13])

        obstacle.write_root_pose_to_sim(torch.cat((positions, orientations), dim=-1), env_ids=env_ids)
        obstacle.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_navigation_target(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    radius_range: tuple[float, float] = (3.0, 4.0),
    angle_range: tuple[float, float] = (-0.8, 0.8),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    marker_name: str | None = None,
    marker_z: float = 0.03,
):
    """Sample a 2D navigation target around the robot at reset."""

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    if not hasattr(env, "_teraffe_target_pos_w"):
        env._teraffe_target_pos_w = torch.zeros(env.scene.num_envs, 3, device=env.device)

    asset = env.scene[asset_cfg.name]
    radius = torch.empty(len(env_ids), device=env.device).uniform_(*radius_range)
    angle = torch.empty(len(env_ids), device=env.device).uniform_(*angle_range)
    local_offset = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle), torch.zeros_like(radius)), dim=-1)
    offset = math_utils.quat_apply(math_utils.yaw_quat(asset.data.root_quat_w[env_ids]), local_offset)[:, :2]

    env._teraffe_target_pos_w[env_ids, :2] = asset.data.root_pos_w[env_ids, :2] + offset
    env._teraffe_target_pos_w[env_ids, 2] = asset.data.root_pos_w[env_ids, 2]

    if marker_name is not None:
        marker: RigidObject = env.scene[marker_name]
        root_state = marker.data.default_root_state[env_ids].clone()
        positions = root_state[:, 0:3]
        positions[:, 0:2] = env._teraffe_target_pos_w[env_ids, 0:2]
        positions[:, 2] = env.scene.env_origins[env_ids, 2] + marker_z
        orientations = root_state[:, 3:7]
        velocities = torch.zeros_like(root_state[:, 7:13])

        marker.write_root_pose_to_sim(torch.cat((positions, orientations), dim=-1), env_ids=env_ids)
        marker.write_root_velocity_to_sim(velocities, env_ids=env_ids)
