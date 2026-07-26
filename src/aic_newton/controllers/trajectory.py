"""Provide validated Cartesian trajectory operations."""

from math import acos, dist, isfinite

import numpy as np

from ..utils.transforms import PoseTuple, normalize_quaternion

MINIMUM_JERK_PEAK_SPEED_SCALE = 15.0 / 8.0


def minimum_jerk(alpha: float) -> float:
    """Evaluate a quintic minimum-jerk interpolation fraction."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)


def interpolate_pose(start: PoseTuple, end: PoseTuple, alpha: float) -> PoseTuple:
    """Interpolate translation and shortest-path orientation."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha == 0.0:
        return start
    if alpha == 1.0:
        return end

    fraction = minimum_jerk(alpha)
    xyz = tuple(start.xyz[axis] + fraction * (end.xyz[axis] - start.xyz[axis]) for axis in range(3))
    start_quat = normalize_quaternion(start.quat_xyzw)
    end_quat = normalize_quaternion(end.quat_xyzw)
    dot_product = sum(left * right for left, right in zip(start_quat, end_quat, strict=True))
    if dot_product < 0.0:
        end_quat = tuple(-component for component in end_quat)
        dot_product = -dot_product
    dot_product = min(1.0, max(-1.0, dot_product))

    if dot_product > 0.9995:
        quaternion = tuple((1.0 - fraction) * start_quat[axis] + fraction * end_quat[axis] for axis in range(4))
        quaternion = normalize_quaternion(quaternion)
    else:
        theta = acos(dot_product)
        sin_theta = np.sin(theta)
        start_weight = np.sin((1.0 - fraction) * theta) / sin_theta
        end_weight = np.sin(fraction * theta) / sin_theta
        quaternion = tuple(float(start_weight * start_quat[axis] + end_weight * end_quat[axis]) for axis in range(4))

    return PoseTuple(xyz, quaternion)


def translation_error(actual: PoseTuple, target: PoseTuple) -> float:
    """Return Euclidean translation error [m]."""
    return dist(actual.xyz, target.xyz)


def orientation_error(actual: PoseTuple, target: PoseTuple) -> float:
    """Return shortest-path orientation error [rad]."""
    actual_quat = normalize_quaternion(actual.quat_xyzw)
    target_quat = normalize_quaternion(target.quat_xyzw)
    dot_product = abs(sum(left * right for left, right in zip(actual_quat, target_quat, strict=True)))
    return 2.0 * acos(min(1.0, max(-1.0, dot_product)))


def validate_motion_segment(
    *,
    distance: float,
    angle: float,
    duration: float,
    max_translation_speed: float,
    max_angular_speed: float,
) -> None:
    """Validate minimum-jerk peak translation and angular segment speeds."""
    values = (distance, angle, duration, max_translation_speed, max_angular_speed)
    if not all(isfinite(value) for value in values):
        raise ValueError("motion segment values must be finite")
    if distance < 0.0 or angle < 0.0:
        raise ValueError("motion segment distance and angle must be non-negative")
    if duration <= 0.0 or max_translation_speed <= 0.0 or max_angular_speed <= 0.0:
        raise ValueError("motion segment duration and speed limits must be positive")

    if MINIMUM_JERK_PEAK_SPEED_SCALE * distance / duration > max_translation_speed + 1.0e-12:
        raise ValueError("translation speed exceeds the state limit")
    if MINIMUM_JERK_PEAK_SPEED_SCALE * angle / duration > max_angular_speed + 1.0e-12:
        raise ValueError("angular speed exceeds the state limit")
