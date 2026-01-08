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
import isaaclab_tasks.manager_based.classic.kanake.mdp.target_path as target_path
# import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
import isaaclab_tasks.manager_based.classic.kanake.mdp as mdp

# Pre-defined configs
from isaaclab_assets.robots.kanake import KANAKE_CFG 
from isaaclab.terrains.config.kanake_rough import ROUGH_TERRAINS_CFG  # isort: skip




@configclass
class MySceneCfg(InteractiveSceneCfg):

    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     # terrain_type="usd",
    #     terrain_type="usd",
    #     # terrain_type="plane",
    #     # terrain_type="generator",
    #     # terrain_generator=KANAKE_RANDOM_TERRAIN_CFG,
    #     # usd_path="/home/hi/IsaacLab_snake/kanake6_sim_523/kanake6_sim/kanake6_sim/urdf/kanake_0610/kanake6_1120_wall.usd",
    #     usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Grid/default_environment.usd",  

    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="average",
    #         restitution_combine_mode="average",
    #         static_friction=0.4,
    #         dynamic_friction=0.4,
    #     ),
    #     debug_vis=False,
    # )

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=0.5,
            dynamic_friction=0.5,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robot
    robot = KANAKE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


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
    # command -> (x,y)
    kanake_command = mdp.KanakeBaseCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(10.0, 15.0), 
        ranges=mdp.KanakeBaseCommandCfg.Ranges(
            pos_x=(-2.0, 2.0),
            pos_y=(-2.0, 2.0),
            # pos_z = (0.05, 0.05),
            heading=(-0.0, 0.0),
        ),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    joint_sine = mdp.JointSineActionCfg(
        asset_name="robot",               
        joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9", "j10", "j11", "j12", "j13", "j14", "j15", "j16"],
        scale=1.0)
    


    

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "kanake_command"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        root_quat_w = ObsTerm(func=mdp.root_quat_w)
        joint_effort = ObsTerm(func=mdp.joint_effort)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True



    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.25, 0.25), "yaw": (0.0,0.0)},
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.5, 0.5), "yaw": (0.0,0.0)},
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
            "position_range": (-np.pi/6, np.pi/6),  
            "velocity_range": (0, 0),
        },
    )

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names = ["Link1", "Link2", "Link3", "Link4","Link5",
            "Link6","Link7", "Link8", "Link9", "Link10", "Link11", "Link12", "Link13", "Link14", "Link15", "tail", "head"]
),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )



    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names = ["Link1", "Link2", "Link3", "Link4","Link5",
            "Link6","Link7", "Link8", "Link9", "Link10", "Link11", "Link12", "Link13", "Link14", "Link15", "tail", "head"]
),
            "mass_distribution_params": (-0.005, 0.005),
            "operation": "add",
        },
    )
    
    


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    ### Task1 - 이동

    # 타겟과의 거리 리워드
    kanake_position_command_error_base = RewTerm(
        func=mdp.kanake_position_command_error_base,
        weight=-1.5,
        params={"command_name": "kanake_command"},
    )
    body_alignment_to_target = RewTerm(
            func=mdp.average_body_velocity_alignment_with_target_pos,
            weight=1.0, 
            params={"command_name": "kanake_command"}
        )
    body_velocity_magnitude = RewTerm(
        func=mdp.average_body_velocity_magnitude,
        weight=0.5,
    )
    # base의 수직 유지(cube)
    upright = RewTerm(func=mdp.kanake_upright_posture_bonus, weight=0.1, params={"threshold": 0.6})

    # # action이 급변하지 않도록 페널티
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    # com_trajectory_logging = RewTerm(
    #     func=mdp.com_trajectory_save,
    #     weight=0.01  
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    max = DoneTerm(func=mdp.root_height_over_maximum, params={"maximum_height": 1.0})

    # bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.57, "asset_cfg": SceneEntityCfg(name="robot")})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    
    # kanake_command_resampling = CurrTerm(
    #     func=mdp.modify_command_resampling_time, 
    #     params={
    #         "command_name": "kanake_command", 
    #         "resampling_time_range": (3.0, 6.0), 
    #         "num_steps": 50000
    #     }
    # )

    terrain_levels = CurrTerm(func=mdp.terrain_levels_pose)


@configclass
class kanakeNavEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=0.2)
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


class kanakeNavEnvCfg_PLAY(kanakeNavEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 1.0
        self.curriculum = None
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # self.commands.kanake_command.resampling_time_range = (10.0,12.0)
        # self.commands.kanake_command.ranges = mdp.KanakeBaseCommandCfg.Ranges(
        #     pos_x=(-1.0, 1.0),    
        #     pos_y=(-1.0, 1.0),     
        #     heading=(0.0, 0.0),  
        # )
        self.commands.kanake_command.resampling_time_range = (1.0e9, 1.0e9) 
        self.commands.kanake_command.debug_vis = False

        self.commands.kanake_command.ranges = mdp.KanakeBaseCommandCfg.Ranges(
            pos_x=(-100.0, 100.0),  
            pos_y=(-100.0, 100.0),  
            heading=(-3.14, 3.14),
        )
        self.episode_length_s = 10000.0
        self.scene.terrain.terrain_type = "usd"
        self.scene.terrain.usd_path = "/home/nuc/IsaacLab_snake/kanake6_sim_523/kanake6_sim/kanake6_sim/urdf/kanake_0610/kanake_navigation.usd"
