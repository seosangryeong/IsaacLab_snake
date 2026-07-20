from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import CUBOID_MARKER_CFG
from isaaclab.utils import configclass

from .navigation_command import TeraffeNavigationCommand


@configclass
class TeraffeNavigationCommandCfg(CommandTermCfg):
    """Configuration for Teraffe 2-D navigation target commands."""

    class_type: type = TeraffeNavigationCommand

    asset_name: str = MISSING
    """Name of the robot asset in the scene."""

    simple_heading: bool = True
    """If True, the commanded heading points from the robot to the sampled target."""

    marker_z: float = 1.0
    """World z position of the debug marker relative to the environment origin."""

    @configclass
    class Ranges:
        """Sampling ranges for the target command."""

        radius: tuple[float, float] = MISSING
        """Target distance from the robot in meters."""

        angle: tuple[float, float] = MISSING
        """Target bearing in the robot yaw frame in radians."""

        heading: tuple[float, float] = (0.0, 0.0)
        """World-frame heading range. Used only when simple_heading is False."""

    ranges: Ranges = MISSING
    """Distribution ranges for the command."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = CUBOID_MARKER_CFG.replace(
        prim_path="/Visuals/Command/teraffe_navigation_goal"
    )
    goal_pose_visualizer_cfg.markers["cuboid"].size = (0.25, 0.25, 0.25)
    goal_pose_visualizer_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.05, 0.85, 0.15)
    )
