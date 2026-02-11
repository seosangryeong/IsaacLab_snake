import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sensors import FrameTransformerCfg, FrameTransformer
from isaaclab.sensors import imu, ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
import math
import numpy as np
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
# from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, CUBOID_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
import torch
from isaaclab.terrains.config.kanake_plane import KANAKE_PLANE_CFG, KANAKE_RANDOM_TERRAIN_CFG, KANAKE_WAVE_TERRATIN_CFG # isort: skip
# import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
import isaaclab_tasks.manager_based.classic.teraffe.mdp as mdp

# Pre-defined configs
from isaaclab_assets.robots.teraffe import TERAFFE_CFG 
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains.config.teraffe_terrain import TERAFFE_TERRAINS_CFG, TERAFFE_WAVE_TERRATIN_CFG





@configclass
class MySceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        # terrain_type="usd",
        # terrain_type="usd",
        # terrain_type="plane",
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",  
        terrain_type="generator",
        terrain_generator=TERAFFE_WAVE_TERRATIN_CFG,
        max_init_terrain_level=5,
        collision_group=-1,

        visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.8, 0.8), 
                metallic=0.0,
                roughness=0.5,
            ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="average",
            static_friction=1.4,
            dynamic_friction=1.2,
        ),
        debug_vis=False,
    )

    # robot
    robot = TERAFFE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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


    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(0.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""


    prismaticjoint = mdp.JointPositionActionCfg(
        asset_name="robot",               
        joint_names=["j1_1","j1_2","j2_1","j2_2","j3_1","j3_2","j4_1","j4_2"],
        scale=1.0)
    
    steerjoint = mdp.JointPositionActionCfg(
        asset_name="robot",               
        joint_names=["j1_steer","j2_steer","j3_steer","j4_steer"],
        scale=1.0)
    
    drivejoint = mdp.JointVelocityActionCfg(
        asset_name="robot",               
        joint_names=["j1_drive","j2_drive","j3_drive","j4_drive"],
        scale=10.0)
    

    

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        base_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        root_quat_w = ObsTerm(func=mdp.root_quat_w)
        # joint_effort = ObsTerm(func=mdp.joint_effort)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        actions = ObsTerm(func=mdp.last_action)





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

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # upright = RewTerm(func=mdp.upright_posture_bonus, weight=2.0, params={"threshold": 1.0})
    upright_shaped = RewTerm(
        func=mdp.upright_posture_shaped_penalty,
        weight=0.5,
        params={"threshold": 0.95} 
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    max = DoneTerm(func=mdp.root_height_over_maximum, params={"maximum_height": 7.0})
    min = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 1.3})
    bad_orientation = DoneTerm(
            func=mdp.bad_orientation, 
            params={
                "limit_angle": 1.05, 
                "asset_cfg": SceneEntityCfg("robot")
            }
        )

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class teraffeEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.0)
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


