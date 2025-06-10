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
from isaaclab.utils import configclass


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
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=0.5,
            dynamic_friction=0.5,
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
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.7, use_default_offset=True)
    # joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[".*"], scale=5.0)
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)

    joint_sine = mdp.JointSineActionCfg(asset_name="robot", joint_names=["j2", "j4", "j6",  "j8",  "j10", "j12", "j14", "j16"])
    # joint_sine_h = mdp.JointSineHorizonActionCfg(
    #     asset_name="robot", 
    #     joint_names=["j2", "j4", "j6",  "j8",  "j10", "j12", "j14", "j16"], 
    #     scale=1.0)
    
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["j1", "j3", "j5", "j7", "j9", "j11", "j13", "j15"], 
        scale= 1.0, 
       )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)

        actions = ObsTerm(func=mdp.last_action)

        

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):

        # base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (0.0, 10.0, 0.0)})
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (0.0, 10.0, 0.0)})
        # # joint_pos = ObsTerm(func=mdp.joint_pos)
        # joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel)
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
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.1, 0.1), "yaw": (0.0,0.0)},
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


#     reset_robot_joints = EventTerm(
#         func=mdp.reset_joints_by_offset,
#         mode="reset",
#         params={
#             "position_range": (-1.0, 1.0),  # -60 ~ 60도 정도
#             "velocity_range": (0, 0),
#         },
#     )

#     physics_material = EventTerm(
#         func=mdp.randomize_rigid_body_material,
#         mode="startup",
#         params={
#             "asset_cfg": SceneEntityCfg("robot", body_names = ["Link1", "Link2", "Link3", "Link4","Link5",
#             "Link6","Link7", "Link8", "Link9", "Link10", "Link11", "Link12", "Link13", "Link14", "Link15", "tail", "head"]
# ),
#             "static_friction_range": (0.1, 1.0),
#             "dynamic_friction_range": (0.1, 1.0),
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

    # progress = RewTerm(func=mdp.progress_reward, weight=15.0, params={"target_pos": (100.0, 0.0, 0.0)})
    # # alive = RewTerm(func=mdp.is_alive, weight=0.5)
    move_to_target = RewTerm(func=mdp.move_to_target_bonus, weight=1.0, params={"threshold": 0.95, "target_pos": (0.0, 10.0, 0.0)})
    # upright = RewTerm(func=mdp.upright_kanake_posture_bonus, weight=2.0, params={"threshold": 0.85})
    BodyLineDistancePenalty = RewTerm(
        func=mdp.BodyLineDistancePenalty,
        weight=-3.0,
        params={"target_pos": (0.0, 10.0, 0.0), "threshold": 0.2}  
    )
    # action_rate_l2 = RewTerm(
    #     func=mdp.action_rate_l2,
    #     weight = -0.01,
    # )
    # progress = RewTerm(func=mdp.progress_reward, weight=20.0, params={"target_pos": (0.0, 10.0, 0.0)})
    # alive = RewTerm(func=mdp.is_alive, weight=0.5)
    # move_to_target = RewTerm(func=mdp.move_to_target_bonus, weight=1.2, params={"threshold": 0.95, "target_pos": (100.0, 0.0, 0.0)})
    upright = RewTerm(func=mdp.upright_posture_shaped, weight=3.0, params={"threshold": 0.8})
    progress_monotonic_reward = RewTerm(
        func=mdp.progress_monotonic_reward,
        weight=10.0,
        params={"target_pos": (0.0, 10.0, 0.0)}
    )
    # BodyOrderReward = RewTerm(
    #     func=mdp.BodyOrderReward,
    #     weight=1.0,
    #     params={"target_pos": (0.0, 10.0, 0.0)}
    # )
    # joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.1)
    # energy = RewTerm(func=mdp.power_consumption, weight=-0.00001, params={"gear_ratio": {".*": 1.0}})
    # ang_vel_0_l2 = RewTerm(func=mdp.ang_vel_0_l2, weight=-0.0003)
    # joint_vel_0 = RewTerm(func=mdp.joint_vel_0, weight=0.01)
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.0003)
    # joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-0.0000003)
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
    #     weight = 0.5,
    #     params={"target_pos": (100.0, 0.0, 0.0)}
    # )
    # joint_limits = RewTerm(
    #     func=mdp.joint_limits_penalty_ratio, weight=-0.1, params={"threshold": 0.80, "gear_ratio": {".*": 1.0}}
    # )
    # HeadTailDistanceReward = RewTerm(
    #     func=mdp.HeadTailDistanceReward,
    #     weight = 0.1,
    #     params={"min_distance": 0.3}
    # )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight = -0.1,
    )
    # joint_limits = RewTerm(
    #     func=mdp.joint_limits_penalty_ratio, weight=-0.1, params={"threshold": 0.99, "gear_ratio": {".*": 1.0}}
    # )
    # joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.1)
    # energy = RewTerm(func=mdp.power_consumption, weight=-0.00001, params={"gear_ratio": {".*": 1.0}})
    # ang_vel_0_l2 = RewTerm(func=mdp.ang_vel_0_l2, weight=-0.0003)
    # joint_vel_0 = RewTerm(func=mdp.joint_vel_0, weight=0.01)
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.0003)
    # joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-0.0000003)
    # lin_vel_x_l2 = RewTerm(func=mdp.lin_vel_x_l2, weight=-0.0003)
    # TorqueEnergyUniformityReward = RewTerm(
    #     func=mdp.TorqueEnergyUniformityReward,
    #     weight = -0.1

    # )
    # head_move_to_target_bonus = RewTerm(
    #     func=mdp.move_to_target_bonus,
    #     weight=0.5,
    #     params={"threshold": 0.98, "target_pos": (100.0, 0.0, 0.0), "asset_cfg": SceneEntityCfg("robot", body_names="head")}
    # )
    # joint_vel_l2_penalty = RewTerm(
    #     func=mdp.joint_vel_l2_penalty,
    #     weight=-0.01,
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
    # HeadTailDistanceReward = RewTerm(
    #     func=mdp.HeadTailDistanceReward,
    #     weight = 0.1,
    #     params={"min_distance": 0.3}
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
    # contact_count = RewTerm(
    #     func=mdp.contact_count_reward,
    #     weight=1.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link.*"),
    #         "force_threshold": 0.6,
    #     },
    # )

    # contact_time = RewTerm(
    #     func=mdp.body_contact_time_reward,
    #     weight=0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link.*"),
    #     },
    # )

    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=0.125,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link.*"),
    #         "threshold": 0.5,
    #     },
    # )


    # continuous_contact_reward = RewTerm(
    #     func=mdp.continuous_contact_reward,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link.*"),
    #     },
    #     weight=1.0
    # )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    max = DoneTerm(func=mdp.root_height_over_maximum, params={"maximum_height": 0.2})
    # bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.57, "asset_cfg": SceneEntityCfg(name="robot")})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass


@configclass
class kanakeEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=0.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # commands: CommandsCfg = CommandsCfg()

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 80.0
        self.sim.render_interval = self.decimation
        # self.sim.render_interval = 2
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 0.5
        self.sim.physics_material.dynamic_friction = 0.5
        self.sim.physics_material.restitution = 0.0
