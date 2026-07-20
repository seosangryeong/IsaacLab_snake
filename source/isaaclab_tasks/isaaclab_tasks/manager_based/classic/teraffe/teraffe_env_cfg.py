import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
# from isaaclab.sensors import TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import math

import isaaclab_tasks.manager_based.classic.teraffe.mdp as mdp

# Pre-defined configs
from isaaclab_assets.robots.teraffe import TERAFFE_CFG 
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG





@configclass
class MySceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        # terrain_type="usd",
        # terrain_type="usd",
        terrain_type="plane",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",  
        # terrain_type="generator",
        # terrain_generator=ROUGH_TERRAINS_CFG,
        # max_init_terrain_level=5,
        collision_group=-1,

        visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.3), 
                metallic=0.0,
                roughness=0.5,
            ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="average",
            static_friction=0.3,
            dynamic_friction=0.4,
        ),
        debug_vis=False,
    )

    # robot
    robot = TERAFFE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Enable this marker only for visual debugging. With 4096 envs it creates 4096 extra rigid objects.
    # target_marker = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/TargetMarker",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 0.03), rot=(1.0, 0.0, 0.0, 0.0)),
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.25, 0.25, 0.5),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.85, 0.15)),
    #     ),
    # )

    # front_depth_camera = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base_link/front_depth_camera",
    #     update_period=0.0,
    #     height=64,
    #     width=64,
    #     data_types=["distance_to_camera"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=18.0,
    #         focus_distance=400.0,
    #         horizontal_aperture=20.955,
    #         clipping_range=(0.03, 5.0),
    #     ),
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(0.45, 0.0, 0.25),
    #         rot=(0.2706, -0.6533, 0.6533, -0.2706),
    #         convention="ros",
    #     ),
    # )

    # obstacle_0 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Obstacle_0",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.0, 1.0, 1.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.18, 0.14)),
    #     ),
    # )

    # obstacle_1 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Obstacle_1",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.0, 0.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.0, 1.0, 1.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.45, 0.75)),
    #     ),
    # )

    # obstacle_2 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Obstacle_2",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 2.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.0, 1.0, 1.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.60, 0.28)),
    #     ),
    # )

    # obstacle_3 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Obstacle_3",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -2.0, 0.5), rot=(1.0, 0.0, 0.0, 0.0)),
    #     spawn=sim_utils.CuboidCfg(
    #         size=(1.0, 1.0, 1.0),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.82, 0.62, 0.15)),
    #     ),
    # )

    # contact_forces = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot3/link[1-4]_drive/collisions", 
    #     history_length=3, 
    #     track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # imu = ImuCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/head"
    # )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""


    navigation_command = mdp.TeraffeNavigationCommandCfg(
        asset_name="robot",
        resampling_time_range=(20.0, 20.0),
        simple_heading=True,
        debug_vis=True,
        ranges=mdp.TeraffeNavigationCommandCfg.Ranges(
            radius=(5.0, 6.0),
            angle=(-math.pi, math.pi),
            heading=(0.0, 0.0),
        ),
    )

    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=0.0,
    #     heading_command=False, 
    #     heading_control_stiffness=0.0,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(0.3, 0.8), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.5, 0.5), heading=(0.0, 0.0)
    #     ),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""


    prismaticjoint = mdp.JointPositionActionCfg(
        asset_name="robot",               
        joint_names=["j1_1","j1_2","j2_1","j2_2","j3_1","j3_2","j4_1","j4_2"],
        scale=0.0,
        offset=0.0,
        use_default_offset=False,
    )
    
    steerjoint = mdp.JointPositionActionCfg(
        asset_name="robot",               
        joint_names=["j1_steer","j2_steer","j3_steer","j4_steer"],
        scale=1.0)
    
    drivejoint = mdp.JointVelocityActionCfg(
        asset_name="robot",               
        joint_names=["j1_drive","j2_drive","j3_drive","j4_drive"],
        scale=8.0)
    

  

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Vector observations for direction-following training."""

        nav_state = ObsTerm(func=mdp.navigation_state)

        # Camera observations. Keep commented while training direction following without vision.
        # front_depth = ObsTerm(
        #     func=mdp.image,
        #     params={"sensor_cfg": SceneEntityCfg("front_depth_camera"), "data_type": "distance_to_camera"},
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (1.57 ,1.57), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.6, 0.6), "roll": (0.0 ,0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},

            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0,0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0, 0),  
            "velocity_range": (0, 0),
        },
    )

    # Navigation targets are generated by CommandsCfg.navigation_command.
    reset_navigation_target = EventTerm(
        func=mdp.reset_navigation_target,
        mode="reset",
        params={"radius_range": (3.0, 4.0)},
        # params={"radius_range": (3.0, 4.0), "marker_name": "target_marker", "marker_z": 0.03},
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="link[1-4]_drive"), 
            "static_friction_range": (1.5, 1.7),  
            "dynamic_friction_range": (1.3, 1.5), 
            "restitution_range": (0.0, 0.0),      
            "num_buckets": 64, 
        },
    )

    # randomize_obstacles = EventTerm(
    #     func=mdp.randomize_obstacle_positions,
    #     mode="reset",
    #     params={
    #         "obstacle_names": ["obstacle_0", "obstacle_1", "obstacle_2", "obstacle_3"],
    #         "active_count_attr": "_teraffe_active_obstacle_count",
    #         "x_range": (-4.0, 4.0),
    #         "y_range": (-4.0, 4.0),
    #         "min_robot_distance": 1.5,
    #         "min_obstacle_distance": 1.25,
    #         "z": 0.5,
    #     },
    # )



#     add_base_mass = EventTerm(
#         func=mdp.randomize_rigid_body_mass,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names = ["Link1", "Link2", "Link3", "Link4","Link5",
#             "Link6","Link7", "Link8", "Link9", "Link10", "Link11", "Link12", "Link13", "Link14", "Link15", "tail", "head"]
# ),
#             "mass_distribution_params": (-2.0, 2.0),
#             "operation": "add",
#         },
#     )
    
    


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # track_lin_vel_xy_exp = RewTerm(
    #     func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    progress_to_target = RewTerm(func=mdp.navigation_progress_velocity, weight=5.0, params={"slow_radius": 1.0})
    face_target = RewTerm(func=mdp.navigation_heading_alignment, weight=0.5)
    forward_velocity_alignment = RewTerm(func=mdp.forward_velocity_alignment, weight=1.0)
    lateral_velocity = RewTerm(func=mdp.lateral_velocity_l2, weight=-3.0)
    backward_velocity = RewTerm(func=mdp.backward_velocity, weight=-2.0)
    stop_near_target = RewTerm(
        func=mdp.navigation_stop_near_target,
        weight=-5.0,
        params={"command_name": "navigation_command", "distance_threshold": 0.5},
    )
    position_tracking = RewTerm(func=mdp.navigation_target_distance_tanh, weight=0.5, params={"std": 2.0})
    position_tracking_fine = RewTerm(func=mdp.navigation_target_distance_tanh, weight=1.0, params={"std": 0.4})
    arrival_bonus = RewTerm(func=mdp.navigation_arrival_bonus, weight=2.0, params={"distance_threshold": 0.35})
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=1.0, params={"threshold": 0.97})
    steer_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["j1_steer", "j2_steer", "j3_steer", "j4_steer"])},
    )
    straight_steer_deviation = RewTerm(
        func=mdp.straight_steer_deviation_l1,
        weight=-0.08,
        params={"command_name": "navigation_command", "lateral_threshold": 0.25, "heading_threshold": 0.25},
    )
    # upright_shaped = RewTerm(
    #     func=mdp.upright_posture_shaped_penalty,
    #     weight=0.5,
    #     params={"threshold": 0.95} 
    # )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # obstacle_too_close = RewTerm(
    #     func=mdp.obstacle_distance_penalty,
    #     weight=-10.0,
    #     params={
    #         "obstacle_names": ["obstacle_0", "obstacle_1", "obstacle_2", "obstacle_3"],
    #         "distance_threshold": 0.9,
    #         "active_count_attr": "_teraffe_active_obstacle_count",
    #     },
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    max = DoneTerm(func=mdp.root_height_over_maximum, params={"maximum_height": 2.5})
    min = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})
    # bad_orientation = DoneTerm(
    #         func=mdp.bad_orientation, 
    #         params={
    #             "limit_angle": 1.05, 
    #             "asset_cfg": SceneEntityCfg("robot")
    #         }
    #     )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    terrain_levels = None
    # obstacle_count = CurrTerm(
    #     func=mdp.obstacle_count_curriculum,
    #     params={
    #         "active_count_attr": "_teraffe_active_obstacle_count",
    #         "initial_count": 0,
    #         "max_count": 4,
    #         "thresholds": (1.4, 1.6, 1.8, 2.0),
    #         "min_steps_between_levels": 2000,
    #         "warmup_resets": 2,
    #     },
    # )


@configclass
class teraffeEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=10.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 80.0
        self.sim.render_interval = 2
        # self.sim.render_interval = 2
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 0.5
        self.sim.physics_material.dynamic_friction = 0.5
        self.sim.physics_material.restitution = 0.0

        self.sim.physx.gpu_max_rigid_contact_count = 2**24  
        self.sim.physx.gpu_max_rigid_patch_count = 2**18  
        self.sim.physx.gpu_heap_capacity = 2**27  
