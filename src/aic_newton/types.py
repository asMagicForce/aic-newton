"""Immutable public types for the ROS-independent simulation engine."""

from dataclasses import dataclass

from .scene.layout import ComponentPlacement, ManipulationFrames, StaticCableAssembly


@dataclass(frozen=True)
class StateSnapshot:
    """One coherent robot snapshot expressed in SI units and ROS ordering."""

    clock_time_s: float
    scene_time_s: float
    joint_names: tuple[str, ...]
    joint_position_rad: tuple[float, ...]
    joint_velocity_rad_s: tuple[float, ...]
    gripper_position_m: tuple[float, float]
    tcp_pose_xyz_xyzw: tuple[float, float, float, float, float, float, float]
    tcp_twist_linear_angular: tuple[float, float, float, float, float, float]
    tcp_wrench_force_torque: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class CameraFrameSet:
    """Three RGB views captured at one simulation timestamp."""

    clock_time_s: float
    width: int
    height: int
    left_rgb: bytes
    center_rgb: bytes
    right_rgb: bytes


@dataclass(frozen=True)
class CableSegmentSnapshot:
    """One Newton capsule segment expressed in world coordinates."""

    pose_xyz_xyzw: tuple[float, float, float, float, float, float, float]
    half_length_m: float
    radius_m: float


@dataclass(frozen=True)
class SceneSnapshot:
    """Task-scene geometry that is not already represented by RobotModel/TF."""

    clock_time_s: float
    board_pose_xyz_xyzw: tuple[float, float, float, float, float, float, float]
    components: tuple[ComponentPlacement, ...]
    static_cables: tuple[StaticCableAssembly, ...]
    manipulation_frames: ManipulationFrames
    cable_points_xyz: tuple[tuple[float, float, float], ...]
    cable_segments: tuple[CableSegmentSnapshot, ...]
    manipulated_object_pose_xyz_xyzw: tuple[float, float, float, float, float, float, float]
    target_pose_xyz_xyzw: tuple[float, float, float, float, float, float, float]


@dataclass(frozen=True)
class ObservationBundle:
    """Robot and scene values captured from one immutable Newton state."""

    state: StateSnapshot
    scene: SceneSnapshot
