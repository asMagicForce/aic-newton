"""Configure robot, manual TCP, and automatic insertion control."""

from dataclasses import dataclass
from math import hypot, isfinite, radians


@dataclass(frozen=True)
class RobotControlConfig:
    """Configure the UR10e and Hand-E position controllers."""

    home_q: tuple[float, ...] = (-0.05494383, -1.04843700, -2.21455836, -1.44928396, 1.57098973, 1.51575613)
    arm_target_ke: tuple[float, ...] = (200.0, 200.0, 200.0, 100.0, 100.0, 100.0)
    arm_target_kd: tuple[float, ...] = (
        56.5685424949,
        56.5685424949,
        56.5685424949,
        21.2132034356,
        21.2132034356,
        21.2132034356,
    )
    arm_effort_limit: tuple[float, ...] = (330.0, 330.0, 150.0, 54.0, 54.0, 54.0)
    gripper_initial_q: float = 0.0073
    gripper_open_q: float = 0.025
    gripper_closed_q: float = 0.0073
    gripper_target_ke: float = 1000.0
    gripper_target_kd: float = 100.0
    gripper_effort_limit: float = 130.0
    tcp_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.172)
    ik_iterations: int = 24
    ik_damping_initial: float = 0.1
    ik_joint_limit_weight: float = 10.0

    def __post_init__(self) -> None:
        """Validate arm dimensions and IK parameters."""
        for name in ("home_q", "arm_target_ke", "arm_target_kd", "arm_effort_limit"):
            if len(getattr(self, name)) != 6:
                raise ValueError(f"{name} must contain six values")
        if self.ik_iterations <= 0 or self.ik_damping_initial <= 0.0 or self.ik_joint_limit_weight <= 0.0:
            raise ValueError("IK iterations, damping, and joint-limit weight must be positive")


@dataclass(frozen=True)
class ManualTCPConfig:
    """Configure keyboard TCP motion and tracking safety."""

    translation_speed_m_s: float = 0.15
    rotation_speed_rad_s: float = radians(45.0)
    max_tracking_error_m: float = 0.2
    stall_timeout_s: float = 2.0
    stall_min_error_m: float = 0.05
    stall_min_speed_m_s: float = 0.002

    def __post_init__(self) -> None:
        """Validate positive manual motion and safety limits."""
        for name in (
            "translation_speed_m_s",
            "rotation_speed_rad_s",
            "max_tracking_error_m",
            "stall_timeout_s",
            "stall_min_error_m",
            "stall_min_speed_m_s",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class MotionProfile:
    """Configure one automatic state duration and speed limit."""

    duration_s: float
    max_translation_speed_m_s: float
    max_rotation_speed_rad_s: float = radians(25.0)

    def __post_init__(self) -> None:
        """Validate positive duration and speed limits."""
        if self.duration_s <= 0.0 or self.max_translation_speed_m_s <= 0.0 or self.max_rotation_speed_rad_s <= 0.0:
            raise ValueError("Motion profile duration and speed limits must be positive")


@dataclass(frozen=True)
class AutomaticInsertionConfig:
    """Configure grasp geometry, safety thresholds, and state motion."""

    nominal_tool_to_sfp_xyz_m: tuple[float, float, float] = (0.0000245241, 0.000900821, 0.166446)
    nominal_tool_to_sfp_quat_xyzw: tuple[float, float, float, float] = (
        -0.822358,
        0.0154996,
        0.0233271,
        -0.568284,
    )
    tcp_translation_tolerance_m: float = 0.005
    tcp_orientation_tolerance_rad: float = radians(5.0)
    grasp_translation_tolerance_m: float = 0.008
    grasp_orientation_tolerance_rad: float = radians(8.0)
    alignment_translation_tolerance_m: float = 0.001
    alignment_orientation_tolerance_rad: float = radians(1.0)
    seat_translation_tolerance_m: float = 0.001
    seat_orientation_tolerance_rad: float = radians(1.0)
    state_timeout_margin: float = 10.0
    gripper_position_tolerance_m: float = 0.001
    # These are relative geometry/path distances, not absolute world heights.
    gripper_pad_center_to_tcp_m: float = 0.033862
    grasp_long_axis_offset_m: float = 0.0
    pregrasp_board_normal_clearance_m: float = 0.08
    sfp_extraction_distance_m: float = 0.033
    preinsert_port_axis_clearance_m: float = 0.10
    align_port_axis_clearance_m: float = 0.005
    trajectory_translation_pause_error_m: float = 0.10
    trajectory_translation_resume_error_m: float = 0.09
    trajectory_orientation_pause_error_rad: float = radians(15.0)
    trajectory_orientation_resume_error_rad: float = radians(10.0)
    trajectory_stall_timeout_s: float = 2.0
    trajectory_translation_min_progress_m_s: float = 0.002
    trajectory_orientation_min_progress_rad_s: float = radians(0.2)
    motion_profiles: tuple[tuple[str, MotionProfile], ...] = (
        ("HOME", MotionProfile(1.0, 0.10)),
        ("MOVE_ABOVE_SFP", MotionProfile(4.0, 0.10)),
        ("DESCEND_TO_GRASP", MotionProfile(2.5, 0.10)),
        ("CLOSE_GRIPPER", MotionProfile(1.0, 0.10)),
        ("EXTRACT_FROM_MOUNT", MotionProfile(2.5, 0.04)),
        ("LIFT_AFTER_EXTRACTION", MotionProfile(2.0, 0.10)),
        ("TRANSFER_ABOVE_PORT", MotionProfile(2.0, 0.10)),
        ("ALIGN_WITH_PORT", MotionProfile(2.5, 0.06)),
        ("INSERT_TO_BOTTOM", MotionProfile(2.5, 0.04)),
        ("OPEN_GRIPPER", MotionProfile(1.0, 0.10)),
        ("RETRACT_FROM_PORT", MotionProfile(2.0, 0.10)),
        ("LIFT_AFTER_RELEASE", MotionProfile(2.0, 0.10)),
    )

    def __post_init__(self) -> None:
        """Validate the nominal grasp transform."""
        if len(self.nominal_tool_to_sfp_xyz_m) != 3:
            raise ValueError("nominal_tool_to_sfp_xyz_m must contain three values")
        if len(self.nominal_tool_to_sfp_quat_xyzw) != 4:
            raise ValueError("nominal_tool_to_sfp_quat_xyzw must contain four values")
        values = (*self.nominal_tool_to_sfp_xyz_m, *self.nominal_tool_to_sfp_quat_xyzw)
        if not all(isfinite(value) for value in values):
            raise ValueError("nominal tool-to-SFP transform must contain only finite values")
        quaternion_norm = hypot(*self.nominal_tool_to_sfp_quat_xyzw)
        if abs(quaternion_norm - 1.0) > 1.0e-5:
            raise ValueError("nominal_tool_to_sfp_quat_xyzw must be normalized")

    def motion_profile(self, state_name: str) -> MotionProfile:
        """Return the configured profile for an automatic state name."""
        try:
            return dict(self.motion_profiles)[state_name]
        except KeyError as error:
            raise ValueError(f"No automatic motion profile configured for {state_name}") from error


ROBOT_CONTROL = RobotControlConfig()
MANUAL_TCP = ManualTCPConfig()
AUTO_INSERTION = AutomaticInsertionConfig()
