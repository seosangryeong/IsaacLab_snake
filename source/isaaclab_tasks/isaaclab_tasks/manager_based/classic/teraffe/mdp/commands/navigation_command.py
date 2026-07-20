from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import TeraffeNavigationCommandCfg


class TeraffeNavigationCommand(CommandTerm):
    """Samples a 2-D navigation target around the robot and reports it in the base yaw frame."""

    cfg: TeraffeNavigationCommandCfg

    def __init__(self, cfg: TeraffeNavigationCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]

        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

        self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "TeraffeNavigationCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Base-frame command for the policy: ``[target_x_b, target_y_b, heading_error]``."""
        return torch.cat((self.pos_command_b[:, :2], self.heading_command_b.unsqueeze(-1)), dim=-1)

    @property
    def world_command_pos(self) -> torch.Tensor:
        """Target position in world frame for reward calculations and visualization."""
        return self.pos_command_w

    def _update_metrics(self):
        self.metrics["error_pos_2d"] = torch.linalg.norm(
            self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=-1
        )
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))

    def _resample_command(self, env_ids: Sequence[int]):
        num_resamples = len(env_ids)
        if num_resamples == 0:
            return

        radius = torch.empty(num_resamples, device=self.device).uniform_(*self.cfg.ranges.radius)
        angle = torch.empty(num_resamples, device=self.device).uniform_(*self.cfg.ranges.angle)
        local_offset = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle), torch.zeros_like(radius)), dim=-1)
        offset_w = quat_apply(yaw_quat(self.robot.data.root_quat_w[env_ids]), local_offset)

        current_pos_w = self.robot.data.root_pos_w[env_ids]
        self.pos_command_w[env_ids, :2] = current_pos_w[:, :2] + offset_w[:, :2]
        self.pos_command_w[env_ids, 2] = self._env.scene.env_origins[env_ids, 2] + self.cfg.marker_z

        if self.cfg.simple_heading:
            target_vec_w = self.pos_command_w[env_ids, :2] - current_pos_w[:, :2]
            self.heading_command_w[env_ids] = torch.atan2(target_vec_w[:, 1], target_vec_w[:, 0])
        else:
            self.heading_command_w[env_ids] = torch.empty(num_resamples, device=self.device).uniform_(
                *self.cfg.ranges.heading
            )

    def _update_command(self):
        target_vec_w = self.pos_command_w - self.robot.data.root_pos_w
        target_vec_w[:, 2] = 0.0
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec_w)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
        elif hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_w,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
        )
