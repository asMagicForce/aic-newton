"""Translate keyboard input into bounded TCP targets."""

import warp as wp

from ..config import MANUAL_TCP


def read_tcp_keyboard(viewer) -> tuple[wp.vec3, wp.vec3, bool]:
    """Read translation, local rotation, and reset commands from a viewer."""
    axis = wp.vec3(
        float(viewer.is_key_down("i")) - float(viewer.is_key_down("k")),
        float(viewer.is_key_down("j")) - float(viewer.is_key_down("l")),
        float(viewer.is_key_down("u")) - float(viewer.is_key_down("o")),
    )
    if viewer.is_key_down("shift"):
        return wp.vec3(0.0, 0.0, 0.0), axis, bool(viewer.is_key_down("r"))
    return axis, wp.vec3(0.0, 0.0, 0.0), bool(viewer.is_key_down("r"))


def integrate_tcp_target(
    target: wp.transform,
    linear_axis: wp.vec3,
    angular_axis: wp.vec3,
    dt: float,
) -> wp.transform:
    """Integrate frame-rate-independent world translation and local rotation."""
    position = wp.transform_get_translation(target)
    rotation = wp.transform_get_rotation(target)
    linear_length = float(wp.length(linear_axis))
    angular_length = float(wp.length(angular_axis))

    if linear_length > 0.0:
        position += linear_axis * (MANUAL_TCP.translation_speed_m_s * dt / linear_length)
    if angular_length > 0.0:
        axis = angular_axis / angular_length
        rotation *= wp.quat_from_axis_angle(axis, MANUAL_TCP.rotation_speed_rad_s * dt)

    return wp.transform(position, wp.normalize(rotation))


def limit_tcp_target_translation(
    target: wp.transform,
    *,
    current_position: wp.vec3,
    max_error: float,
) -> wp.transform:
    """Project a TCP target into a translation-error safety bound."""
    target_position = wp.transform_get_translation(target)
    error = target_position - current_position
    error_length = float(wp.length(error))
    if error_length <= max_error:
        return target
    limited_position = current_position + error * (max_error / error_length)
    return wp.transform(limited_position, wp.transform_get_rotation(target))


def update_translation_stall(
    *,
    linear_axis: wp.vec3,
    current_position: wp.vec3,
    previous_position: wp.vec3,
    target_position: wp.vec3,
    elapsed: float,
    blocked: bool,
    dt: float,
) -> tuple[float, bool]:
    """Latch a translation command that cannot make progress."""
    axis_length = float(wp.length(linear_axis))
    if axis_length <= 1.0e-8:
        return 0.0, False
    if blocked:
        return elapsed, True

    direction = linear_axis / axis_length
    progress_speed = float(wp.dot(current_position - previous_position, direction)) / dt
    tracking_error = float(wp.length(target_position - current_position))
    if tracking_error >= MANUAL_TCP.stall_min_error_m and progress_speed < MANUAL_TCP.stall_min_speed_m_s:
        elapsed += dt
    else:
        elapsed = 0.0
    return elapsed, elapsed >= MANUAL_TCP.stall_timeout_s - 1.0e-9
