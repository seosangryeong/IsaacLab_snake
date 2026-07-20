# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from dataclasses import field

from . import binary_joint_actions, joint_actions, modular_actions, joint_actions_to_limits, non_holonomic_actions, task_space_actions, sine_actions, sine_h_actions, sine_v_actions, cpg_actions, sine_amp_actions,sine_actions_hold 
import numpy as np
##
# Joint actions.
##


@configclass
class JointActionCfg(ActionTermCfg):
    """Configuration for the base joint action term.

    See :class:`JointAction` for more details.
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""
    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""


@configclass
class JointPositionActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions.JointPositionAction

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """


@configclass
class RelativeJointPositionActionCfg(JointActionCfg):
    """Configuration for the relative joint position action term.

    See :class:`RelativeJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions.RelativeJointPositionAction

    use_zero_offset: bool = True
    """Whether to ignore the offset defined in articulation asset. Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to zero.
    """


@configclass
class JointVelocityActionCfg(JointActionCfg):
    """Configuration for the joint velocity action term.

    See :class:`JointVelocityAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions.JointVelocityAction

    use_default_offset: bool = True
    """Whether to use default joint velocities configured in the articulation asset as offset.
    Defaults to True.

    This overrides the settings from :attr:`offset` if set to True.
    """


@configclass
class JointEffortActionCfg(JointActionCfg):
    """Configuration for the joint effort action term.

    See :class:`JointEffortAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions.JointEffortAction

    


##
# Joint actions rescaled to limits.
##


@configclass
class JointPositionToLimitsActionCfg(ActionTermCfg):
    """Configuration for the bounded joint position action term.

    See :class:`JointPositionToLimitsAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions_to_limits.JointPositionToLimitsAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""

    rescale_to_limits: bool = True
    """Whether to rescale the action to the joint limits. Defaults to True.

    If True, the input actions are rescaled to the joint limits, i.e., the action value in
    the range [-1, 1] corresponds to the joint lower and upper limits respectively.

    Note:
        This operation is performed after applying the scale factor.
    """


@configclass
class EMAJointPositionToLimitsActionCfg(JointPositionToLimitsActionCfg):
    """Configuration for the exponential moving average (EMA) joint position action term.

    See :class:`EMAJointPositionToLimitsAction` for more details.
    """

    class_type: type[ActionTerm] = joint_actions_to_limits.EMAJointPositionToLimitsAction

    alpha: float | dict[str, float] = 1.0
    """The weight for the moving average (float or dict of regex expressions). Defaults to 1.0.

    If set to 1.0, the processed action is applied directly without any moving average window.
    """


##
# Gripper actions.
##


@configclass
class BinaryJointActionCfg(ActionTermCfg):
    """Configuration for the base binary joint action term.

    See :class:`BinaryJointAction` for more details.
    """

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    open_command_expr: dict[str, float] = MISSING
    """The joint command to move to *open* configuration."""
    close_command_expr: dict[str, float] = MISSING
    """The joint command to move to *close* configuration."""


@configclass
class BinaryJointPositionActionCfg(BinaryJointActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`BinaryJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = binary_joint_actions.BinaryJointPositionAction


@configclass
class BinaryJointVelocityActionCfg(BinaryJointActionCfg):
    """Configuration for the binary joint velocity action term.

    See :class:`BinaryJointVelocityAction` for more details.
    """

    class_type: type[ActionTerm] = binary_joint_actions.BinaryJointVelocityAction


##
# Non-holonomic actions.
##


@configclass
class NonHolonomicActionCfg(ActionTermCfg):
    """Configuration for the non-holonomic action term with dummy joints at the base.

    See :class:`NonHolonomicAction` for more details.
    """

    class_type: type[ActionTerm] = non_holonomic_actions.NonHolonomicAction

    body_name: str = MISSING
    """Name of the body which has the dummy mechanism connected to."""
    x_joint_name: str = MISSING
    """The dummy joint name in the x direction."""
    y_joint_name: str = MISSING
    """The dummy joint name in the y direction."""
    yaw_joint_name: str = MISSING
    """The dummy joint name in the yaw direction."""
    scale: tuple[float, float] = (1.0, 1.0)
    """Scale factor for the action. Defaults to (1.0, 1.0)."""
    offset: tuple[float, float] = (0.0, 0.0)
    """Offset factor for the action. Defaults to (0.0, 0.0)."""


##
# Task-space Actions.
##


@configclass
class DifferentialInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for inverse differential kinematics action term.

    See :class:`DifferentialInverseKinematicsAction` for more details.
    """

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type: type[ActionTerm] = task_space_actions.DifferentialInverseKinematicsAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    body_name: str = MISSING
    """Name of the body or frame for which IK is performed."""
    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""
    scale: float | tuple[float, ...] = 1.0
    """Scale factor for the action. Defaults to 1.0."""
    controller: DifferentialIKControllerCfg = MISSING
    """The configuration for the differential IK controller."""


@configclass
class OperationalSpaceControllerActionCfg(ActionTermCfg):
    """Configuration for operational space controller action term.

    See :class:`OperationalSpaceControllerAction` for more details.
    """

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type: type[ActionTerm] = task_space_actions.OperationalSpaceControllerAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""

    body_name: str = MISSING
    """Name of the body or frame for which motion/force control is performed."""

    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""

    task_frame_rel_path: str = None
    """The path of a ``RigidObject``, relative to the sub-environment, representing task frame. Defaults to None."""

    controller_cfg: OperationalSpaceControllerCfg = MISSING
    """The configuration for the operational space controller."""

    position_scale: float = 1.0
    """Scale factor for the position targets. Defaults to 1.0."""

    orientation_scale: float = 1.0
    """Scale factor for the orientation (quad for ``pose_abs`` or axis-angle for ``pose_rel``). Defaults to 1.0."""

    wrench_scale: float = 1.0
    """Scale factor for the wrench targets. Defaults to 1.0."""

    stiffness_scale: float = 1.0
    """Scale factor for the stiffness commands. Defaults to 1.0."""

    damping_ratio_scale: float = 1.0
    """Scale factor for the damping ratio commands. Defaults to 1.0."""

    nullspace_joint_pos_target: str = "none"
    """The joint targets for the null-space control: ``"none"``, ``"zero"``, ``"default"``, ``"center"``.

    Note: Functional only when ``nullspace_control`` is set to ``"position"`` within the
        ``OperationalSpaceControllerCfg``.
    """

@configclass
class JointSineActionCfg(JointActionCfg):
    class_type: type[ActionTerm] = sine_actions.JointSineAction
    preserve_order: bool = True

    # action_scale_amph: float = 15.0      # 수평 진폭(deg) 조절 범위: ±15도
    # action_scale_ampv: float = 5.0       # 수직 진폭(deg) 조절 범위: ±5도
    # action_scale_wavefqh: float = 0.2    # 수평 주파수(Hz) 조절 범위: ±0.2Hz
    # action_scale_wavefqv: float = 0.4    # 수직 주파수(Hz) 조절 범위: ±0.4Hz
    # action_scale_shapefqh: float = 0.5   # 수평 위상 계수 조절 범위: ±0.5
    # action_scale_shapefqv: float = 0.5   # 수직 위상 계수 조절 범위: ±0.5
    additional_joint_scale: float = 0.0



@configclass
class JointSineAmpActionCfg(JointActionCfg):
    class_type: type[ActionTerm] = sine_amp_actions.JointSineAmpAction
    preserve_order: bool = True

    clip_ranges: list[tuple[float, float]] = [


        (0.3, 0.6),  
        (0.5, 1.2),    
        (0.5, 1.0),   
        (np.pi/4, np.pi/2),  
        (0.3, 0.6),  
        (0.5, 1.2),    
        (0.5, 1.0),   
        (np.pi/4, np.pi/2),  


    ]

    # min_joint_scale: float = 0.5
    # enable_additional_joint_values: bool = True
    additional_joint_scale: float = 1.0


@configclass
class JointSineHorizonActionCfg(JointActionCfg):
    class_type: type[ActionTerm] = sine_h_actions.JointSineHorizonAction
    preserve_order: bool = True

    clip_ranges: list[tuple[float, float]] = [

        # # horizontal
        (0.5, 1.0),  # amplitude
        (0.5, 1.0),  # frequency
        (np.pi/2, np.pi/2),  # phase

        # horizontal
        # (0.5, 1.5),  # amplitude
        # (0.5, 1.5),  # frequency
        # (0.4, 1.2),  # phase


    ]


@configclass
class JointSineVerticalActionCfg(JointActionCfg):
    class_type: type[ActionTerm] = sine_v_actions.JointSineVerticalAction
    preserve_order: bool = True

    clip_ranges: list[tuple[float, float]] = [

        # vertical
        (0.6, 0.8),  # amplitude
        (0.8, 1.0),  # frequency
        (np.pi/2, np.pi/2),  # phase
    ]

@configclass
class JointCPGActionCfg(JointActionCfg):

    # 사용할 ActionTerm 클래스 지정
    class_type: type[ActionTerm] = cpg_actions.JointCPGAction

    joint_names_horz: list[str] = MISSING 
    joint_names_vert: list[str] = MISSING

    # 3. CPG 하이퍼파라미터
    a: float = 3.0      # 진폭 수렴 속도
    mu: float = 1.0     # 관절 간 커플링(동기화) 계수
    rl_policy_update_period_s: float = 1.0

    # 4. 출력 스케일
    output_scale: float = 1.0

    # 5. RL 액션(7차원) 범위 설정
    # 순서: [R_horz, R_vert, omega(주파수), theta_horz, theta_vert, delta_horz, delta_vert]
    action_scale: list[float] = [
            0.75,   # R_horz:   변화폭 0.75 -> 최종 범위 [0.0, 1.5]
            0.75,   # R_vert:   변화폭 0.75 -> 최종 범위 [0.0, 1.5]
            0.1,    # omega:    변화폭 [-0.1, 0.1]
            3.14,   # theta_h:  변화폭 pi -> 최종 범위 [-pi, pi]
            3.14,   # theta_v:  변화폭 pi -> 최종 범위 [-pi, pi]
            0.3,    # delta_h:  (유지) 이전 설정인 0.3rad(약 17도)는 회전에 적합
            0.1     # delta_v:  변화폭 0.1 -> 최종 범위 [-0.1, 0.1]
        ]    
    # action_min: list[float] = [0.0, 0.0, -np.pi, -np.pi, -np.pi, -0.5, -0.5]
    # # action_max: list[float] = [1.0, 1.0, np.pi*2, np.pi/4, np.pi/4, 0.0, 0.0]
    # action_max: list[float] = [0.87, 0.17, np.pi, 0.9, 1.8, 0.0, 0.0]


@configclass
class JointSineHoldActionCfg(JointActionCfg):
    class_type: type[ActionTerm] = sine_actions_hold.JointSineHoldAction
    preserve_order: bool = True

    clip_ranges: list[tuple[float, float]] = [

        (0.5, 0.7),  # amplitude
        (1.0, 1.2),  # frequency
        (np.pi/2, np.pi/2),  # phase

        # horizontal

        (1.0, 1.2),  # amplitude
        (0.5, 0.7),  # frequency
        (np.pi/3, np.pi/3),  # phase
        ]    
    enable_additional_joint_values: bool = False
    additional_joint_scale: float = 1.0


@configclass
class ModularJointEffortActionCfg(ActionTermCfg):
    """
    모듈러 관절 토크 제어를 위한 설정(Configuration) 클래스입니다.

    COMPOSER 논문과 같이 각 에이전트가 자신의 관절 토크를 직접 제어하는
    다개체 RL 방식을 지원합니다.
    """

    # 이 설정을 사용할 때 인스턴스화할 실제 구현 클래스를 지정합니다.
    class_type: type[ActionTerm] = modular_actions.ModularJointEffortAction

    # -- 액션에 필요한 파라미터들 --

    asset_name: str = "robot"
    joint_names: list[str] = MISSING
    num_agents: int = MISSING
    action_dim_per_agent: int = MISSING
    scale: float = 1.0
