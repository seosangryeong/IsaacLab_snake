# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause



from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
import os


##
# Configuration
##

TERAFFE_ASSET_DIR = os.path.join(os.path.dirname(__file__), "teraffe")
TERAFFE_USD_PATH = os.environ.get("TERAFFE_USD_PATH", os.path.join(TERAFFE_ASSET_DIR, "teraffe.usd"))
if not os.path.exists(TERAFFE_USD_PATH):
    legacy_usd_path = os.path.abspath("./teraffe/teraffe.usd")
    if os.path.exists(legacy_usd_path):
        TERAFFE_USD_PATH = legacy_usd_path


TERAFFE_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TERAFFE_USD_PATH,
        activate_contact_sensors=True,
        # rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #     disable_gravity=False,
        #     max_linear_velocity=100.0,
        #     max_angular_velocity=100.0,
        #     max_depenetration_velocity=10.0,
        # ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=16,
        ),       
        copy_from_source=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.4),
        # pos=(0.0, 0.0, 2.0),
        # rot=(0.70710678, 0.70710678, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),

    actuators = {
        # prismatic: j1_1/2 ~ j4_1/2
        "prismatic_pd": ImplicitActuatorCfg(
            joint_names_expr=[r"j[1-4]_[12]"],   # j1_1, j1_2, ..., j4_2
            stiffness=5000.0,
            damping=500.0,
        ),

        # steer: j1_steer ~ j4_steer
        "steer_pd": ImplicitActuatorCfg(
            joint_names_expr=[r"j[1-4]_steer"],
            stiffness=20.0,
            damping=10.0,
        ),

        # drive: j1_drive ~ j4_drive (velocity control 쓸 거면 게인 0)
        "drive_vel": ImplicitActuatorCfg(
            joint_names_expr=[r"j[1-4]_drive"],
            stiffness=0.0,
            damping=60.0,
            # effort_limit=30.0,      
            # effort_limit_sim=30.0,
        ),
    },

)
