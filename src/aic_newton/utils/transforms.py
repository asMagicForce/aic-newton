"""Provide validated pose and quaternion operations."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import asin, atan2, copysign, cos, hypot, isfinite, pi, sin, sqrt

import warp as wp


@dataclass(frozen=True)
class PoseTuple:
    """Describe a position and XYZW orientation."""

    xyz: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float]


def transform_from_row(row: Sequence[float]) -> wp.transform:
    """Convert one body-transform array row to a Warp transform."""
    return wp.transform(
        wp.vec3(float(row[0]), float(row[1]), float(row[2])),
        wp.quat(float(row[3]), float(row[4]), float(row[5]), float(row[6])),
    )


def pose_from_transform(transform: wp.transform) -> PoseTuple:
    """Convert a Warp transform to an immutable pose."""
    return PoseTuple(
        (float(transform[0]), float(transform[1]), float(transform[2])),
        (float(transform[3]), float(transform[4]), float(transform[5]), float(transform[6])),
    )


def transform_from_pose(pose: PoseTuple) -> wp.transform:
    """Convert an immutable pose to a Warp transform."""
    return wp.transform(wp.vec3(*pose.xyz), wp.quat(*pose.quat_xyzw))


def transform_from_components(
    xyz: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float],
) -> wp.transform:
    """Build a Warp transform from tuple components."""
    return wp.transform(wp.vec3(*xyz), wp.quat(*quat_xyzw))


def normalize_quaternion(
    quaternion: tuple[float, float, float, float],
    *,
    field: str = "quaternion",
) -> tuple[float, float, float, float]:
    """Validate and normalize an XYZW quaternion."""
    if not all(isfinite(component) for component in quaternion):
        raise ValueError(f"{field} must contain only finite values")
    norm = hypot(*quaternion)
    if not isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{field} must have finite non-zero length")
    return tuple(component / norm for component in quaternion)


def normalize_direction(
    direction: tuple[float, float, float],
    *,
    field: str,
) -> tuple[float, float, float]:
    """Validate and normalize a direction vector."""
    if not all(isfinite(component) for component in direction):
        raise ValueError(f"{field} must contain only finite values")
    norm = hypot(*direction)
    if not isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{field} must have finite non-zero length")
    return tuple(component / norm for component in direction)


def quat_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Multiply two XYZW quaternions."""
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def quat_rotate(
    quaternion: tuple[float, float, float, float],
    xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a vector by an XYZW quaternion."""
    x, y, z, w = normalize_quaternion(quaternion)
    vx, vy, vz = xyz
    twice_cross = (2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx))
    return (
        vx + w * twice_cross[0] + y * twice_cross[2] - z * twice_cross[1],
        vy + w * twice_cross[1] + z * twice_cross[0] - x * twice_cross[2],
        vz + w * twice_cross[2] + x * twice_cross[1] - y * twice_cross[0],
    )


def rpy_quaternion(rpy: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Convert fixed-axis roll, pitch, and yaw to XYZW."""
    roll, pitch, yaw = (angle * 0.5 for angle in rpy)
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quaternion_rpy(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Convert an XYZW unit quaternion to fixed-axis roll, pitch, and yaw."""
    x, y, z, w = normalize_quaternion(quaternion)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = copysign(pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else asin(sin_pitch)
    return (
        atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
        pitch,
        atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
    )


def compose_components(
    parent_xyz: tuple[float, float, float],
    parent_quat: tuple[float, float, float, float],
    child_xyz: tuple[float, float, float],
    child_quat: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Compose parent and child pose components."""
    parent_quat = normalize_quaternion(parent_quat)
    child_quat = normalize_quaternion(child_quat)
    rotated = quat_rotate(parent_quat, child_xyz)
    return (
        tuple(parent_xyz[axis] + rotated[axis] for axis in range(3)),
        normalize_quaternion(quat_multiply(parent_quat, child_quat)),
    )


def pose_tuple(
    xyz: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
    *,
    field: str,
) -> PoseTuple:
    """Build a finite pose with a normalized orientation."""
    if not all(isfinite(component) for component in xyz):
        raise ValueError(f"{field}.xyz must contain only finite values")
    return PoseTuple(xyz, normalize_quaternion(quaternion, field=f"{field}.quat_xyzw"))


def compose_pose(parent: PoseTuple, child: PoseTuple) -> PoseTuple:
    """Compose two poses."""
    xyz, quaternion = compose_components(parent.xyz, parent.quat_xyzw, child.xyz, child.quat_xyzw)
    return PoseTuple(xyz, quaternion)


def inverse_pose(pose: PoseTuple) -> PoseTuple:
    """Invert a pose."""
    x, y, z, w = normalize_quaternion(pose.quat_xyzw)
    inverse_quaternion = (-x, -y, -z, w)
    return PoseTuple(
        quat_rotate(inverse_quaternion, tuple(-component for component in pose.xyz)),
        inverse_quaternion,
    )


def quaternion_from_axes(
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    z_axis: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Construct an XYZW quaternion from orthonormal rotation-matrix columns."""
    m00, m10, m20 = x_axis
    m01, m11, m21 = y_axis
    m02, m12, m22 = z_axis
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = 2.0 * sqrt(trace + 1.0)
        quaternion = ((m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale, 0.25 * scale)
    elif m00 > m11 and m00 > m22:
        scale = 2.0 * sqrt(1.0 + m00 - m11 - m22)
        quaternion = (0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale, (m21 - m12) / scale)
    elif m11 > m22:
        scale = 2.0 * sqrt(1.0 + m11 - m00 - m22)
        quaternion = ((m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale, (m02 - m20) / scale)
    else:
        scale = 2.0 * sqrt(1.0 + m22 - m00 - m11)
        quaternion = ((m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale, (m10 - m01) / scale)
    return normalize_quaternion(quaternion)
