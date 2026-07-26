"""Load and shape the reviewed flexible-cable reference."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path

import numpy as np
import warp as wp

from ..config import AUTO_INSERTION, CABLE
from ..utils.transforms import quat_rotate as _quat_rotate
from .robot import AIC_TOOL_TO_GRIPPER_TCP


@dataclass(frozen=True)
class CableReference:
    """Describe the official zero-deformation AIC cable."""

    max_segment_length_m: float
    centerline: tuple[tuple[float, float, float], ...]
    mount_to_cable_xyz: tuple[float, float, float]
    mount_to_cable_quat_xyzw: tuple[float, float, float, float]
    lc_plug_xyz: tuple[float, float, float]
    lc_plug_quat_xyzw: tuple[float, float, float, float]
    sfp_module_xyz: tuple[float, float, float]
    sfp_module_quat_xyzw: tuple[float, float, float, float]
    sc_plug_xyz: tuple[float, float, float]
    sc_plug_quat_xyzw: tuple[float, float, float, float]


def _finite_tuple(value: object, *, length: int, field: str) -> tuple[float, ...]:
    """Validate a fixed-size finite numeric JSON array."""
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} values")
    result = tuple(float(item) for item in value)
    if not all(isfinite(item) for item in result):
        raise ValueError(f"{field} must contain only finite values")
    return result


def _resample_centerline(
    points: tuple[tuple[float, float, float], ...],
    max_segment_length: float,
) -> tuple[tuple[float, float, float], ...]:
    """Resample a complete polyline into equal-length segments."""
    segment_lengths: list[float] = []
    distances = [0.0]
    for start, end in zip(points, points[1:]):
        length = sum((end[axis] - start[axis]) ** 2 for axis in range(3)) ** 0.5
        if length <= 1.0e-8:
            raise ValueError("centerline segments must have positive length")
        segment_lengths.append(length)
        distances.append(distances[-1] + length)

    segment_count = max(1, ceil(distances[-1] / max_segment_length - 1.0e-9))
    sample_distances = tuple(sample_index * distances[-1] / segment_count for sample_index in range(segment_count + 1))
    result: list[tuple[float, float, float]] = []
    source_segment = 0
    for distance in sample_distances:
        while source_segment + 1 < len(segment_lengths) and distance > distances[source_segment + 1]:
            source_segment += 1
        fraction = (distance - distances[source_segment]) / segment_lengths[source_segment]
        start = points[source_segment]
        end = points[source_segment + 1]
        result.append(tuple(start[axis] + fraction * (end[axis] - start[axis]) for axis in range(3)))
    return tuple(result)


def _blend_centerline_start_tangent(
    points: tuple[tuple[float, float, float], ...],
    tangent: tuple[float, float, float],
    *,
    blend_length: float,
    max_segment_length: float,
) -> tuple[tuple[float, float, float], ...]:
    """Blend a prescribed start tangent back into an existing polyline."""
    distances = [0.0]
    for start, end in zip(points, points[1:]):
        distances.append(distances[-1] + sum((end[axis] - start[axis]) ** 2 for axis in range(3)) ** 0.5)

    join_index = next(
        (index for index, distance in enumerate(distances) if distance >= blend_length),
        len(points) - 2,
    )
    join_index = min(max(join_index, 2), len(points) - 2)
    start = points[0]
    end = points[join_index]
    join_direction = tuple(points[join_index + 1][axis] - points[join_index - 1][axis] for axis in range(3))
    join_length = sum(component * component for component in join_direction) ** 0.5
    join_tangent = tuple(component / join_length for component in join_direction)
    chord_length = sum((end[axis] - start[axis]) ** 2 for axis in range(3)) ** 0.5
    handle_length = 0.4 * chord_length
    control_start = tuple(start[axis] + handle_length * tangent[axis] for axis in range(3))
    control_end = tuple(end[axis] - handle_length * join_tangent[axis] for axis in range(3))
    sample_count = max(8, ceil(blend_length / max_segment_length) * 4)
    blend = []
    for sample in range(sample_count + 1):
        fraction = sample / sample_count
        inverse = 1.0 - fraction
        blend.append(
            tuple(
                inverse**3 * start[axis]
                + 3.0 * inverse**2 * fraction * control_start[axis]
                + 3.0 * inverse * fraction**2 * control_end[axis]
                + fraction**3 * end[axis]
                for axis in range(3)
            )
        )
    return (*blend[:-1], *points[join_index:])


def _add_reference_endpoint_strain_reliefs(
    points: tuple[tuple[float, float, float], ...],
    *,
    lc_plug_quat_xyzw: tuple[float, float, float, float],
    sc_plug_quat_xyzw: tuple[float, float, float, float],
    length: float,
    blend_length: float,
    max_segment_length: float,
) -> tuple[tuple[float, float, float], ...]:
    """Add straight connector-normal sections at both cable endpoints."""
    lc_axis = _quat_rotate(lc_plug_quat_xyzw, (0.0, -1.0, 0.0))
    sc_axis = _quat_rotate(sc_plug_quat_xyzw, (-1.0, 0.0, 0.0))
    lc_relief_end = tuple(points[0][axis] + length * lc_axis[axis] for axis in range(3))
    sc_relief_end = tuple(points[-1][axis] + length * sc_axis[axis] for axis in range(3))
    interior = (lc_relief_end, *points[1:-1], sc_relief_end)
    interior = tuple(
        point
        for index, point in enumerate(interior)
        if index == 0 or sum((point[axis] - interior[index - 1][axis]) ** 2 for axis in range(3)) ** 0.5 > 1.0e-8
    )
    interior = _blend_centerline_start_tangent(
        interior,
        lc_axis,
        blend_length=blend_length,
        max_segment_length=max_segment_length,
    )
    interior = tuple(
        reversed(
            _blend_centerline_start_tangent(
                tuple(reversed(interior)),
                sc_axis,
                blend_length=blend_length,
                max_segment_length=max_segment_length,
            )
        )
    )
    return (points[0], *_resample_centerline(interior, max_segment_length), points[-1])


def load_cable_reference(path: Path) -> CableReference:
    """Load and validate the vendored AIC cable reference."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("AIC cable reference must use schema version 1")

    source_points = data.get("source_centerline")
    if not isinstance(source_points, list) or len(source_points) < 3:
        raise ValueError("source_centerline must contain at least three points")
    points = tuple(
        _finite_tuple(point, length=3, field=f"source_centerline[{index}]") for index, point in enumerate(source_points)
    )

    source_segment_length = float(data.get("max_segment_length_m", 0.0))
    if not isfinite(source_segment_length) or source_segment_length <= 0.0:
        raise ValueError("max_segment_length_m must be finite and positive")
    max_segment_length = CABLE.segment_length_m

    transforms = data.get("transforms")
    if not isinstance(transforms, dict):
        raise ValueError("transforms must be an object")

    def transform(name: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        value = transforms.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"transforms.{name} must be an object")
        xyz = _finite_tuple(value.get("xyz"), length=3, field=f"transforms.{name}.xyz")
        quat = _finite_tuple(value.get("quat_xyzw"), length=4, field=f"transforms.{name}.quat_xyzw")
        norm = sum(component * component for component in quat) ** 0.5
        if abs(norm - 1.0) > 5.0e-4:
            raise ValueError(f"transforms.{name}.quat_xyzw must be normalized")
        return xyz, tuple(component / norm for component in quat)

    mount_xyz, mount_quat = transform("mount_to_cable")
    lc_xyz, lc_quat = transform("cable_to_lc_plug")
    sfp_xyz, sfp_quat = transform("cable_to_sfp_module")
    sc_xyz, sc_quat = transform("cable_to_sc_plug")

    def source_field(name: str) -> str:
        value = data.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
        return value

    for name in ("source_repository", "source_ref", "source_commit", "source_model"):
        source_field(name)

    return CableReference(
        max_segment_length_m=max_segment_length,
        centerline=_resample_centerline(points, max_segment_length),
        mount_to_cable_xyz=mount_xyz,
        mount_to_cable_quat_xyzw=mount_quat,
        lc_plug_xyz=lc_xyz,
        lc_plug_quat_xyzw=lc_quat,
        sfp_module_xyz=sfp_xyz,
        sfp_module_quat_xyzw=sfp_quat,
        sc_plug_xyz=sc_xyz,
        sc_plug_quat_xyzw=sc_quat,
    )


def _world_without_original_cable(source: str) -> str:
    """Remove the AIC cable-plugin subtree and its cross-file constraints."""
    root = ET.fromstring(source)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("AIC world MJCF has no <worldbody> element")

    removed_names: set[str] = set()
    for body in list(worldbody):
        if body.tag != "body" or body.get("name") != "cable_end_0":
            continue
        removed_names.update(child.get("name") for child in body.iter("body") if child.get("name") is not None)
        worldbody.remove(body)

    if "cable_end_0" not in removed_names:
        raise ValueError("AIC world MJCF has no cable_end_0 body")

    contact = root.find("contact")
    if contact is not None:
        for exclude in list(contact):
            if exclude.get("body1") in removed_names or exclude.get("body2") in removed_names:
                contact.remove(exclude)
        if not list(contact):
            root.remove(contact)

    equality = root.find("equality")
    if equality is not None:
        # AIC's world equality section contains the cross-file tool-to-plug
        # weld. The native rod installs that attachment explicitly below.
        root.remove(equality)

    return ET.tostring(root, encoding="unicode")


def _attachment_filter_segment_count(points: np.ndarray, filter_length: float) -> int:
    """Count cable segments beginning within an attachment exclusion length."""
    points = np.asarray(points, dtype=np.float32)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    segment_starts = np.concatenate(([0.0], np.cumsum(segment_lengths[:-1], dtype=np.float64)))
    return int(np.count_nonzero(segment_starts < filter_length))


def _body_poses_by_suffix(
    labels: list[str],
    poses: list[wp.transform],
    suffixes: tuple[str, ...],
) -> dict[str, wp.transform]:
    """Collect uniquely matching body poses by label suffix."""
    if len(labels) != len(poses):
        raise ValueError("Body labels and poses must have matching lengths")

    result: dict[str, wp.transform] = {}
    for suffix in suffixes:
        matches = [pose for label, pose in zip(labels, poses, strict=True) if label.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one body label ending in {suffix!r}, found {len(matches)}")
        result[suffix] = matches[0]
    return result


def _geom_transform(source: str, name: str) -> wp.transform:
    """Read one MJCF geom transform using MuJoCo's WXYZ quaternion order."""
    geom = ET.fromstring(source).find(f".//geom[@name='{name}']")
    if geom is None:
        raise ValueError(f"MJCF has no geom named {name!r}")
    position = [float(value) for value in geom.get("pos", "0 0 0").split()]
    quaternion = [float(value) for value in geom.get("quat", "1 0 0 0").split()]
    if len(position) != 3 or len(quaternion) != 4:
        raise ValueError(f"MJCF geom {name!r} has an invalid transform")
    return wp.transform(
        wp.vec3(*position),
        wp.quat(quaternion[1], quaternion[2], quaternion[3], quaternion[0]),
    )


def _mounted_sfp_grasp_tcp_target(mounted_lc_pose: wp.transform) -> wp.transform:
    """Return the TCP pose for the configured nominal tool-to-SFP grasp."""
    nominal_tool_to_sfp = wp.transform(
        wp.vec3(*AUTO_INSERTION.nominal_tool_to_sfp_xyz_m),
        wp.quat(*AUTO_INSERTION.nominal_tool_to_sfp_quat_xyzw),
    )
    target_tool_pose = mounted_lc_pose * wp.transform_inverse(nominal_tool_to_sfp)
    return target_tool_pose * AIC_TOOL_TO_GRIPPER_TCP
