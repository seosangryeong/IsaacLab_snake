import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import math
from isaaclab.sensors import FrameTransformer

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

# Pre-defined configs
from isaaclab_assets.robots.kanake import KANAKE_CFG 

@configclass
class MySceneCfg(InteractiveSceneCfg):

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.7,
            dynamic_friction=0.4,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot
    robot = KANAKE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )




@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=0.05)
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)
    # joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*"], scale=5.0)
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.6, use_default_offset=True)

    # joint_sine = mdp.JointSineActionCfg(asset_name="robot", joint_names=[".*"])
    joint_sine_position = mdp.JointSinePositionActionCfg(
        asset_name="robot",
        joint_names=[".*"]
    )
    # joint_sine_h = mdp.JointSineHorizonActionCfg(
    #     asset_name="robot", 
    #     joint_names=["j2", "j4", "j6",  "j8",  "j10", "j12", "j14", "j16"], 
    #     scale=1.0)
    
    # joint_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=["j1", "j3", "j5", "j7", "j9", "j11", "j13", "j15"], 
    #     scale=0.5, 
    #     use_default_offset=True)
    
    # joint_sine_v = mdp.JointSineVerticalActionCfg(
    #     asset_name="robot", 
    #     joint_names=["j1", "j3", "j5", "j7", "j9", "j11", "j13", "j15"], 
    #     scale=1.0)
    
    # joint_pos = mdp.JointPositionActionCfg(
    #     asset_name="robot", 
    #     joint_names=["j2", "j4", "j6",  "j8",  "j10", "j12", "j14", "j16"], 
    #     scale=1.0, 
    #     use_default_offset=True)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # base_height = ObsTerm(func=mdp.base_pos_z)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (100.0, 0.0, 0.0)})
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (100.0, 0.0, 0.0)})
        # projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_pos_to_target = ObsTerm(func=mdp.base_pos_to_target, params={"target_pos": (100.0, 0.0, 0.0)})
        # base_quat_w = ObsTerm(func=mdp.root_quat_w)
        # heading_w = ObsTerm(func=mdp.heading_w)
        # base_forward_vector = ObsTerm(func=mdp.base_forward_vector)
        # body_pos = ObsTerm(func=mdp.body_pos, flatten_history_dim=True)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        actions = ObsTerm(func=mdp.last_action)

        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            # self.history_length = 3

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-1.57,1.57)},
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.1, 0.1),"roll": (0.0, 0.0), "pitch": (0.0,0.0), "yaw": (-1.57,1.57)},
            # "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.05, 0.05),"roll": (0.0, 0.0), "pitch": (0.0,0.0), "yaw": (0.0, 0.0)},

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
            "position_range": (-0.5, 0.5),  # -60 ~ 60도 정도
            "velocity_range": (0, 0),
        },
    )

#     physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names = ["Link1", "Link2", "Link3", "Link4","Link5",
#             "Link6","Link7", "Link8", "Link9", "Link10", "Link11", "Link12", "Link13", "Link14", "Link15", "tail", "head"]
# ),
#             "static_friction_range": (0.4, 0.6),
#             "dynamic_friction_range": (0.3, 0.5),
#             "restitution_range": (0.0, 0.0),
#             "num_buckets": 64,
#         },
#     )

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

    progress = RewTerm(func=mdp.progress_reward, weight=5.0, params={"target_pos": (100.0, 0.0, 0.0)})
    terminated = RewTerm(func=mdp.is_terminated, weight=-1.0)
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight = -0.01
    )
    # action_l2 = RewTerm(
    #     func=mdp.action_l2,
    #     weight = -0.0001
    # )
    # LocalWorldAlignmentReward = RewTerm(
    #     func=mdp.LocalWorldAlignmentReward,
    #     weight=0.01
    # )
    # body_height_penalty = RewTerm(
    #     func=mdp.body_height_penalty,
    #     weight=-0.3
    # )
    # alive = RewTerm(func=mdp.is_alive, weight=0.5)
    """command"""
    # track_lin_vel_xy_exp = RewTerm(
    #     func=mdp.track_lin_vel_xy_exp, 
    #     weight=1.0, 
    #     params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_exp, 
    #     weight=1.0, 
    #     params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    move_to_target = RewTerm(func=mdp.move_to_target_bonus, weight=1.0, params={"threshold": 0.9, "target_pos": (100.0, 0.0, 0.0)})
    upright = RewTerm(func=mdp.upright_posture_bonus, weight=2.0, params={"threshold": 0.9})
    # upright_penalty = RewTerm(func=mdp.upright_posture_penalty, weight=0.5, params={"threshold": 1.0})
    # debug = RewTerm(func=mdp.debug, weight=1.0) #디버깅용
    # heading = RewTerm(func=mdp.heading, weight=2.0, params={"target_pos": (100.0, 0.0, 0.0)})
    # joint_nomove_penalty = RewTerm(func=mdp.joint_nomove_penalty,params={"threshold": 2.0}, weight=-0.5) #2라디안
    # base_up_proj1 = RewTerm(func=mdp.base_up_proj1, weight=0.1)
    # BodyOrderReward = RewTerm(func=mdp.BodyOrderReward, weight=1.0, params={"target_pos": (100.0, 0.0, 0.0)})
    # joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.1)
    # energy = RewTerm(func=mdp.power_consumption, weight=-0.0001, params={"gear_ratio": {".*": 1.0}})
    # ang_vel_0_l2 = RewTerm(func=mdp.ang_vel_0_l2, weight=-0.0003)
    # joint_vel_0 = RewTerm(func=mdp.joint_vel_0, weight=0.01)
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.0003)
    # joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-0.0000001)
    # lin_vel_x_l2 = RewTerm(func=mdp.lin_vel_x_l2, weight=-0.0003)
    # TorqueEnergyUniformityReward = RewTerm(
    #     func=mdp.TorqueEnergyUniformityReward,
    #     weight = -0.1

    # )


    # distancereward = RewTerm(
    #     func=mdp.DistanceReward, 
    #     weight=-0.6,  
    #     params={"threshold": 0.2}
    # )

    # linealignmentreward = RewTerm(
    #     func=mdp.LineAlignmentReward,
    #     weight = 1.0,
    #     params={"target_pos": (100.0, 0.0, 0.0)}
    # )

    # joint_limits = RewTerm(
    #     func=mdp.joint_limits_penalty_ratio, weight=-0.1, params={"threshold": 0.80, "gear_ratio": {".*": 1.0}}
    # )

    # HeadTailDistancePenalty = RewTerm(
    #     func=mdp.HeadTailDistancePenalty,
    #     weight = 1.0,
    #     params={"min_distance": 0.2}
    # )

    # balanced_body_contact_reward = RewTerm(
    #     func = mdp.balanced_body_contact_reward,
    #     weight = 0.5,
    #     params={
    #          "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"),
    #          "force_threshold": 0.7,
    #          "balance_threshold": 0.2
    #     }   
    # )



# @configclass
# class CommandsCfg:

#     base_velocity = mdp.UniformVelocityCommandCfg(
#         asset_name="robot",
#         resampling_time_range=(8.0, 8.0),
#         rel_standing_envs=0.0,
#         rel_heading_envs=1.0,
#         heading_command=False, # true : 목표방향과 현재 방향의 차이로 회전속도 계산, false : z축 회전 속도 랜덤 샘플링. yaw를 유지하는게 false.
#         # heading_control_stiffness=0.5, #heading command가 true일 때 사용 : 회전속도 결정
#         debug_vis=True,
#         ranges=mdp.UniformVelocityCommandCfg.Ranges(
#             lin_vel_x=(-0.2, 0.2), 
#             lin_vel_y=(-0.2, 0.2), 
#             ang_vel_z=(-0.2, 0.2), 
#             # heading=(-math.pi, math.pi) #이것도 heading command가 ture일 때 사용
#         ),
#     )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    max = DoneTerm(func=mdp.root_height_over_maximum, params={"maximum_height": 0.2})
    # bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.57, "asset_cfg": SceneEntityCfg(name="robot")})


@configclass
class CurriculumCfg:
    # action_rate_l2 = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "action_rate_l2", "weight": -0.005, "num_steps": 5000}
    # )

    # heading = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "heading", "weight": 2.0, "num_steps": 8000}
    # )    
    joint_acc_l2 = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_acc_l2", "weight": -0.000001, "num_steps": 10000}
    )



@configclass
class kanakeEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.5
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.05
