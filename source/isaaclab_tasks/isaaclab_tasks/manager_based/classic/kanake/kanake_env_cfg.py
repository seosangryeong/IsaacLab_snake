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

# TARGET_MARKER_CFG = FRAME_MARKER_CFG.replace(prim_path="/World/target_marker")
# arrow_cfg = GREEN_ARROW_X_MARKER_CFG.replace(
#     prim_path="/World/my_green_arrow",
#     markers={
#         k: v.replace(scale=(1.0, 1.0, 2.0)) for k, v in GREEN_ARROW_X_MARKER_CFG.markers.items()
#     }
# )
# TARGET_BOX = CUBOID_MARKER_CFG.replace( prim_path="/World/target_box")
# box_cfg = CUBOID_MARKER_CFG.replace(
#     prim_path="/World/target_box",
#     markers={
#         "cuboid": CUBOID_MARKER_CFG.markers["cuboid"].replace(
#             size=(0.1, 0.1, 0.1),
#         )
#     }
# )




@configclass
class MySceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        # terrain_type="usd",
        terrain_type="plane",
        # terrain_type="generator",
        # terrain_generator=KANAKE_RANDOM_TERRAIN_CFG,
        # usd_path="/home/hi/IsaacLab_snake/kanake6_sim_523/kanake6_sim/kanake6_sim/urdf/kanake_0610/kanake6_1120_wall.usd",
        # terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=0.6,
            dynamic_friction=0.6,
        ),
        # visual_material=sim_utils.PreviewSurfaceCfg(
        #     # diffuse_color=(0.065, 0.0725, 0.080),#회색
        #     diffuse_color=(1.0, 1.0, 1.0),
        #     emissive_color=(0.0, 0.0, 0.0),
        #     roughness= 0.5,
        #     metallic = 0.3,
        #     opacity = 1.0
        # ),
        debug_vis=False,
    )

    # robot
    robot = KANAKE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/head/Camera",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn= None,
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )
    # camera = FrameTransformerCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/head/Camera",  
    #     target_frames=[],  
    # )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    # command -> (x,y,z)
    kanake_command = mdp.KanakeBaseCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(10.0, 10.0), 
        ranges=mdp.KanakeBaseCommandCfg.Ranges(
            pos_x=(-1.0, 1.0),
            pos_y=(-1.0, 1.0),
            # pos_z = (0.05, 0.05),
            heading=(-0.0, 0.0),
        ),
        debug_vis=True,
    )

    head_command = mdp.KanakeWorldCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0), 
        ranges=mdp.KanakeWorldCommandCfg.Ranges(

            pos_z=(0.15, 0.15),

            pitch=(-0.1, 0.1),
            yaw=(0.0, 0.0),
            roll=(0.0, 0.0)
        ),
        debug_vis=True,
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    # joint_effort = mdp.JointEffortActionCfg(
    #     asset_name="robot", 
    #     joint_names=[".*"], 
    #     scale=0.2,
    #     clip={".*": (-5.0, 5.0)}
    #     )
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
    # joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*"], scale=5.0)
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["j1", "j2", "j3"], scale=0.5)
    # joint_sine_hold = mdp.JointSineHoldActionCfg(
    #     asset_name="robot",
    #     joint_names=[".*"]
    # )
    joint_sine = mdp.JointSineActionCfg(asset_name="robot", 
                                        joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7", "j8", "j9", "j10", "j11", "j12", "j13", "j14", "j15", "j16"],
                                        scale=1.0)

    # joint_cpg = mdp.JointCPGActionCfg(asset_name="robot", joint_names=[".*"],scale=1.0)

    # joint_sine_amp = mdp.JointSineAmpActionCfg(
    #     asset_name="robot",
    #     joint_names=[".*"]
    # )
    # joint_sine_h = mdp.JointSineHorizonActionCfg(
    #     asset_name="robot", 
    #     joint_names=["j2", "j4", "j6",  "j8",  "j10", "j12", "j14", "j16"], 
    #     scale=1.0)
    
    # joint_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=["j2", "j4", "j6",  "j8",  "j10", "j12", "j14", "j16"], 
    #     scale= 1.0, 
    #    )
    # joint_pos_v = mdp.JointSineVerticalActionCfg(
    #     asset_name="robot",
    #     joint_names=["j1", "j3", "j5", "j7", "j9", "j11", "j13", "j15"],
    #     scale=1.0,
    # )
    

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""
        # joint_pos = ObsTerm(func=mdp.joint_pos)
        # joint_vel = ObsTerm(func=mdp.joint_vel)
        # joint_effort = ObsTerm(func=mdp.joint_effort)
        # pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "kanake_command"})

        # actions = ObsTerm(func=mdp.last_action)
        # base_height = ObsTerm(func=mdp.base_pos_z)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        # base_pos = ObsTerm(func=mdp.base_pos)
        joint_effort = ObsTerm(func=mdp.joint_effort)

        # base_angle_to_target_command = ObsTerm(func=mdp.base_angle_to_target_command, params={"command_name": "kanake_command"})
        # base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (5.0, 0.0, 0.0)})
        # # joint_pos = ObsTerm(func=mdp.joint_pos)
        # joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "kanake_command"})
        head_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_command"})
        # image_features = ObsTerm(
        #     func=mdp.image_features,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("Camera"),  
        #         "data_type": "rgb",
        #         "model_name": "resnet18",  
        #         "model_device": "cuda",   
        #     }
        # )
        # cube_pose_w = ObsTerm(func=mdp.cube_pose_w)

        joint_vel = ObsTerm(func=mdp.joint_vel)
        # joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        joint_pos = ObsTerm(func=mdp.joint_pos)
        # joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)

        # joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        actions = ObsTerm(func=mdp.last_action)





        

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):

        base_height = ObsTerm(func=mdp.base_pos_z)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_pos = ObsTerm(func=mdp.base_pos)
        joint_effort = ObsTerm(func=mdp.joint_effort)

        # base_angle_to_target_command = ObsTerm(func=mdp.base_angle_to_target_command, params={"command_name": "kanake_command"})
        # base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (5.0, 0.0, 0.0)})
        # # joint_pos = ObsTerm(func=mdp.joint_pos)
        # joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "kanake_command"})
        head_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_command"})
        # cube_pose_w = ObsTerm(func=mdp.cube_pose_w)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        # joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        joint_pos = ObsTerm(func=mdp.joint_pos)
        # joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True



    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-1.57,1.57)},
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.25, 0.25), "yaw": (-np.pi,np.pi)},
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

#     add_base_mass = EventTerm(
#         func=mdp.randomize_rigid_body_mass,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names = ["Link1", "Link2", "Link3", "Link4","Link5",
#             "Link6","Link7", "Link8", "Link9", "Link10", "Link11", "Link12", "Link13", "Link14", "Link15", "tail", "head"]
# ),
#             "mass_distribution_params": (-0.005, 0.005),
#             "operation": "add",
#         },
#     )
    
    


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    ### Task1 - 이동

    # 타겟과의 거리 리워드
    kanake_position_command_error_base = RewTerm(
        func=mdp.kanake_position_command_error_base,
        weight=-2.0,
        params={"command_name": "kanake_command"},
    )

    # 타겟과의 거리 리워드(타겟과 가까워질수록 리워드를 더 크게(tanh))
    # kanake_position_command_error_tanh = RewTerm(
    #     func=mdp.kanake_position_command_error_tanh,
    #     weight=50.0,
    #     params={"std": 0.05, "command_name": "kanake_command"},
    # )

    # 타겟까지의 벡터와 베이스 속도벡터의 유사도
    # velocity_target_alignment_reward = RewTerm(
    #     func=mdp.velocity_target_alignment_reward,
    #     weight=1.0,
    #     params={"command_name": "kanake_command"},
    # )

    #몸체의 평균 xy벡터와 base to target 벡터의 유사도
    # average_body_velocity_alignment_reward = RewTerm(
    #     func=mdp.average_body_velocity_alignment_reward,
    #     weight=1.0,
    #     params={"command_name": "kanake_command"},
    # )
    # speed_towards_target_reward = RewTerm(
    #     func=mdp.speed_towards_target_reward,
    #     weight=0.1,
    #     params={"command_name": "kanake_command"}
    # )

    ####################################
    
    ### Task2 - head가 타겟을 보도록

    # cube(head)의 z좌표와 kanake_world_command의 z좌표 커맨드 오차
    # cube_height_penalty = RewTerm(
    #     func=mdp.cube_height_penalty,
    #     weight=-50.0,
    #     params={"command_name": "head_command"}
    # )

    # head의 z방향 속도 페널티
    # head_vertical_velocity_penalty = RewTerm(func=mdp.head_vertical_velocity_penalty, weight=-50.0)

    # cube(head)의 로컬 x축 방향과 base-target 벡터의 유사도
    # cube_x_axis_target_alignment_reward = RewTerm(
    #     func=mdp.cube_x_axis_target_alignment_reward,
    #     weight=10.0,
    #     params={"command_name": "kanake_command"},
    # )

    ####################################

    ### 자세 유지

    # base의 수직 유지(cube)
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=1.0, params={"threshold": 0.9})

    # 몸체가 타겟방향으로 순서대로 배치될 수 있도록(앞을 향해 갈수 있도록)
    # BodyOrderReward = RewTerm(func=mdp.BodyOrderReward,weight=1.0)

    # action이 급변하지 않도록 페널티
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # head 로컬 x직선과 body들의 거리 합
    # BaseXAxisDistanceReward = RewTerm(
    #     func=mdp.BaseXAxisDistanceReward, 
    #     weight=-1.0, 
    #     params={"threshold": 0.1}
    # )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # max = DoneTerm(func=mdp.root_height_over_maximum, params={"maximum_height": 0.5})

    # bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.57, "asset_cfg": SceneEntityCfg(name="robot")})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    
    # camera_orientation_alignment_reward = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "camera_orientation_alignment_reward", "weight": 2.0, "num_steps": 10000}
    # )
    # cube_direction_alignment_reward = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "cube_direction_alignment_reward", "weight": 2.0, "num_steps": 10000}
    # )
    # cube_height_reward = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "cube_height_reward", "weight": 3.0, "num_steps": 10000}
    # )
    # head_vertical_velocity_penalty = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "head_vertical_velocity_penalty", "weight": -1.0, "num_steps": 10000}
    # )
    # cube_orientation_reward = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "cube_orientation_reward", "weight": 1.0, "num_steps": 10000}
    # )
    # action_rate_l2 = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate_l2", "weight": -0.001, "num_steps": 10000}
    # )
    # speed_towards_target_reward = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "speed_towards_target_reward", "weight": 2.0, "num_steps": 10000}
    # )


@configclass
class kanakeEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=0.0)
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
        self.sim.render_interval = self.decimation
        # self.sim.render_interval = 2
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 0.5
        self.sim.physics_material.dynamic_friction = 0.5
        self.sim.physics_material.restitution = 0.0

        self.sim.physx.gpu_max_rigid_contact_count = 2**24  
        self.sim.physx.gpu_max_rigid_patch_count = 2**18  
        self.sim.physx.gpu_heap_capacity = 2**27  


class kanakeEnvCfg_PLAY(kanakeEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
