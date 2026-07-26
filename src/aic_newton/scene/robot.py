"""Configure the UR5e, Hand-E gripper, and TCP kinematics."""

import newton
import warp as wp

from ..config import ROBOT_CONTROL
from ..utils.labels import find_label_index as _find_label_index

AIC_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
AIC_GRIPPER_JOINT_NAMES = (
    "gripper/left_finger_joint",
    "gripper/right_finger_joint",
)
AIC_GRIPPER_TCP_OFFSET = wp.vec3(*ROBOT_CONTROL.tcp_offset_m)
AIC_TOOL_TO_GRIPPER_TCP = wp.transform(AIC_GRIPPER_TCP_OFFSET, wp.quat_identity())


def _arm_coordinate_indices(model) -> list[int]:
    """Return the six arm coordinate indices in kinematic order."""
    starts = model.joint_q_start.numpy() if hasattr(model.joint_q_start, "numpy") else model.joint_q_start
    return [int(starts[_find_label_index(model.joint_label, name)]) for name in AIC_ARM_JOINT_NAMES]


def _configure_mujoco_gravity_compensation(builder: newton.ModelBuilder, bodies: list[int]) -> None:
    """Enable full MuJoCo gravity compensation for selected bodies."""
    gravcomp = builder.custom_attributes["mujoco:gravcomp"]
    if gravcomp.values is None:
        gravcomp.values = {}
    for body in bodies:
        gravcomp.values[body] = 1.0


def _set_robot_home(
    builder: newton.ModelBuilder,
) -> None:
    """Set the robot home pose and position-control gains."""
    if len(ROBOT_CONTROL.arm_target_ke) != 6 or len(ROBOT_CONTROL.arm_target_kd) != 6:
        raise ValueError("Arm target gains must contain six values")
    for name, value, target_ke, target_kd, effort_limit in zip(
        AIC_ARM_JOINT_NAMES,
        ROBOT_CONTROL.home_q,
        ROBOT_CONTROL.arm_target_ke,
        ROBOT_CONTROL.arm_target_kd,
        ROBOT_CONTROL.arm_effort_limit,
        strict=True,
    ):
        joint = _find_label_index(builder.joint_label, name)
        coord = builder.joint_q_start[joint]
        dof = builder.joint_qd_start[joint]
        builder.joint_q[coord] = value
        builder.joint_target_q[coord] = value
        builder.joint_target_ke[dof] = target_ke
        builder.joint_target_kd[dof] = target_kd
        builder.joint_effort_limit[dof] = effort_limit
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION_VELOCITY)

    for name in AIC_GRIPPER_JOINT_NAMES:
        joint = _find_label_index(builder.joint_label, name)
        coord = builder.joint_q_start[joint]
        dof = builder.joint_qd_start[joint]
        builder.joint_q[coord] = ROBOT_CONTROL.gripper_initial_q
        builder.joint_target_q[coord] = ROBOT_CONTROL.gripper_initial_q
        builder.joint_target_ke[dof] = ROBOT_CONTROL.gripper_target_ke
        builder.joint_target_kd[dof] = ROBOT_CONTROL.gripper_target_kd
        builder.joint_effort_limit[dof] = ROBOT_CONTROL.gripper_effort_limit
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION)
