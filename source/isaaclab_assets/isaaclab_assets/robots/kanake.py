# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause



from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.managers import SceneEntityCfg
import os


##
# Configuration
##


KANAKE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
# /home/nuc/IsaacLab_snake/kanake6_sim_523/kanake6_sim_0808_4.SLDASM/urdf/
#/home/nuc/IsaacLab_snake/kanake6_sim_523/kanake6_sim/kanake6_sim/urdf/kanake_0610/
        usd_path="./kanake6_sim_523/kanake6_sim/kanake6_sim/urdf/kanake_0610/kanake_0806.usd",
        # usd_path="./kanake6_sim_523/kanake6_sim_0814/kanake6_0814_flat.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # retain_accelerations=False,
            # linear_damping=0.0,
            # angular_damping=0.0,
            max_linear_velocity=10.0,
            max_angular_velocity=10.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            # sleep_threshold=0.005,
            # stabilization_threshold=0.001,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     collision_enabled = True
        # ),
    
            
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),

    actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=["j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9", "j10", "j11", "j12", "j13", "j14", "j15", "j16"],
                stiffness =0.5,
                damping =0.1,
                # stiffness = 10.0,
                # damping =5.0,
                effort_limit = 10.0, #Nm
                effort_limit_sim = 10.0, #Nm
                # velocity_limit = 5.7, #rad/s

            ),
        },
)

