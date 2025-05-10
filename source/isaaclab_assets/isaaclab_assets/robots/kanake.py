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


##
# Configuration
##
# /home/smarthc/Downloads/kanake6_sim/kanake6_sim/urdf/kanake_0610/
KANAKE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/hi/Downloads/kanake6_sim/kanake6_sim/urdf/kanake_0610/kanake6_1120.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=5,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled = True
        ),
    
            
        copy_from_source=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "j1": 0.0,
            "j2": 0.0,
            "j3": 0.0,
            "j4": 0.0,
            "j5": 0.0,
            "j6": 0.0,
            "j7": 0.0,
            "j8": 0.0,
            "j9": 0.0,
            "j10": 0.0,
            "j11": 0.0,
            "j12": 0.0,
            "j13": 0.0,
            "j14": 0.0,
            "j15": 0.0,
            "j16": 0.0,
        },
        joint_vel={".*": 0.0},
    ),

    actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=["j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9", "j10", "j11", "j12", "j13", "j14", "j15", "j16"],
                stiffness = 10.0,
                damping = 6.0,
                # stiffness = 10.0,
                # damping =5.0,
                effort_limit = 2.0, #Nm
                velocity_limit = 3.0, #rad/s

            ),
        },
)

