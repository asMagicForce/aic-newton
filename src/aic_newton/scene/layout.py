# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Define the deterministic AIC task-board scene."""

from dataclasses import dataclass
from math import cos, pi, sin
from pathlib import Path
from xml.etree import ElementTree

from ..config import CABLE, SCENE
from ..utils.transforms import (
    PoseTuple,
)
from ..utils.transforms import (
    compose_components as _compose_pose,
)
from ..utils.transforms import (
    compose_pose as _compose_pose_tuple,
)
from ..utils.transforms import (
    inverse_pose as _inverse_pose,
)
from ..utils.transforms import (
    normalize_direction as _normalize_direction,
)
from ..utils.transforms import (
    normalize_quaternion as _normalize_quaternion,
)
from ..utils.transforms import (
    pose_tuple as _pose_tuple,
)
from ..utils.transforms import (
    quat_rotate as _quat_rotate,
)
from ..utils.transforms import (
    quaternion_rpy as _quaternion_rpy,
)
from ..utils.transforms import (
    rpy_quaternion as _rpy_quaternion,
)
from .cable import (
    CableReference,
)
from .cable import (
    _add_reference_endpoint_strain_reliefs as _add_endpoint_strain_reliefs,
)


@dataclass(frozen=True)
class CablePairPlacement:
    """Describe one paired SFP and SC mount placement."""

    index: int
    rail: int
    translation: float


@dataclass(frozen=True)
class ComponentPlacement:
    """Describe one component pose in world coordinates."""

    name: str
    asset_name: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


@dataclass(frozen=True)
class SceneLayout:
    """Describe a complete deterministic AIC board layout."""

    board_xyz: tuple[float, float, float]
    board_rpy: tuple[float, float, float]
    cable_pairs: tuple[CablePairPlacement, ...]
    nic_translations: tuple[float, ...]
    sc_port_translations: tuple[float, ...]


@dataclass(frozen=True)
class StaticCableAssembly:
    """Describe one static cable assembly in world coordinates."""

    name: str
    centerline: tuple[tuple[float, float, float], ...]
    cable_xyz: tuple[float, float, float]
    lc_plug_xyz: tuple[float, float, float]
    lc_plug_quat_xyzw: tuple[float, float, float, float]
    sfp_module_xyz: tuple[float, float, float]
    sfp_module_quat_xyzw: tuple[float, float, float, float]
    sc_plug_xyz: tuple[float, float, float]
    sc_plug_quat_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class SceneHandles:
    """Group indexed shapes belonging to the static AIC scene."""

    base_shapes: tuple[int, ...]
    sfp_mount_shapes: tuple[tuple[int, ...], ...]
    sc_port_shapes: tuple[tuple[int, ...], ...]
    nic_card_shapes: tuple[tuple[int, ...], ...]
    cable_assemblies: tuple[StaticCableAssembly, ...]
    cable_shapes: dict[str, list[int]]


@dataclass(frozen=True)
class ManipulationFrames:
    """Describe the reviewed frames for one automatic insertion."""

    cable_name: str
    sfp_module: PoseTuple
    port_bottom: PoseTuple
    port_entrance: PoseTuple
    board_normal: tuple[float, float, float]
    mount_extraction_axis: tuple[float, float, float]


SFP_RAIL_ORIGINS = ((0.01, -0.10625, 0.01), (0.01, 0.10625, 0.01))
# The reviewed cable curve was captured with the SFP fixtures in these slots.
CABLE_REFERENCE_SFP_RAIL_ORIGINS = ((0.055, -0.10625, 0.01), (0.055, 0.10625, 0.01))
SC_RAIL_ORIGINS = ((0.1, -0.10625, 0.012), (0.0985, 0.10625, 0.01))
NIC_RAIL_Y = (-0.1745, -0.1345, -0.0945, -0.0545, -0.0145)
SC_PORT_ROW_Y = (0.0295, 0.0295, 0.0295, 0.0705, 0.0705)
SFP_FACE_OFFSET_M = 0.02365
# AIC maps the raw SDF port frame to the policy frame by Rx(+90 deg), then
# maps the policy port frame to the SFP module by Rx(180 deg).
SFP_TARGET_FROM_RAW_PORT_RPY = (-pi / 2.0, 0.0, 0.0)


def default_layout() -> SceneLayout:
    """Return the reviewed deterministic AIC layout."""
    return SceneLayout(
        board_xyz=SCENE.board_xyz_m,
        board_rpy=SCENE.board_rpy_rad,
        cable_pairs=tuple(
            CablePairPlacement(index=index, rail=rail, translation=translation)
            for index, (rail, translation) in enumerate(
                zip(SCENE.cable_pair_rails, SCENE.cable_pair_translations_m, strict=True)
            )
        ),
        nic_translations=SCENE.nic_translations_m,
        sc_port_translations=SCENE.sc_port_translations_m,
    )


def _sdf_link_pose(root: ElementTree.Element, link_name: str) -> PoseTuple:
    """Read one named link pose from the reviewed NIC SDF."""
    link = root.find(f".//link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"NIC SDF must define link '{link_name}'")
    pose_text = link.findtext("pose")
    if pose_text is None:
        raise ValueError(f"NIC SDF link '{link_name}' must define a pose")
    try:
        values = tuple(float(value) for value in pose_text.split())
    except ValueError as error:
        raise ValueError(f"NIC SDF link '{link_name}' pose must contain numeric values") from error
    if len(values) != 6:
        raise ValueError(f"NIC SDF link '{link_name}' pose must contain exactly 6 values")
    return _pose_tuple(
        (values[0], values[1], values[2]),
        _rpy_quaternion((values[3], values[4], values[5])),
        field=f"NIC SDF link '{link_name}'",
    )


def build_static_cable_assemblies(
    layout: SceneLayout,
    reference: CableReference,
) -> tuple[StaticCableAssembly, ...]:
    """Build the five static AIC cable assemblies."""
    placements = {placement.name: placement for placement in component_placements(layout)}
    assemblies: list[StaticCableAssembly] = []
    identity = (0.0, 0.0, 0.0, 1.0)
    for pair in layout.cable_pairs:
        mount = placements[f"sfp_mount_{pair.index}"]
        mount_quat = _rpy_quaternion(mount.rpy)
        cable_xyz, cable_quat = _compose_pose(
            mount.xyz,
            mount_quat,
            reference.mount_to_cable_xyz,
            reference.mount_to_cable_quat_xyzw,
        )

        def connector_pose(
            xyz: tuple[float, float, float],
            quat: tuple[float, float, float, float],
        ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
            return _compose_pose(cable_xyz, cable_quat, xyz, quat)

        lc_xyz, lc_quat = connector_pose(reference.lc_plug_xyz, reference.lc_plug_quat_xyzw)
        sfp_xyz, sfp_quat = connector_pose(reference.sfp_module_xyz, reference.sfp_module_quat_xyzw)
        sc_xyz, sc_quat = connector_pose(reference.sc_plug_xyz, reference.sc_plug_quat_xyzw)
        reference_lc_to_sfp = _compose_pose_tuple(
            _inverse_pose(PoseTuple(lc_xyz, lc_quat)),
            PoseTuple(sfp_xyz, sfp_quat),
        )
        sfp_pose = _compose_pose_tuple(
            PoseTuple(mount.xyz, mount_quat),
            PoseTuple(SCENE.mount_to_sfp_xyz_m, SCENE.mount_to_sfp_quat_xyzw),
        )
        lc_pose = _compose_pose_tuple(sfp_pose, _inverse_pose(reference_lc_to_sfp))
        lc_xyz, lc_quat = lc_pose.xyz, lc_pose.quat_xyzw
        sfp_xyz, sfp_quat = sfp_pose.xyz, sfp_pose.quat_xyzw
        centerline = tuple(_compose_pose(cable_xyz, cable_quat, point, identity)[0] for point in reference.centerline)
        reference_origin = CABLE_REFERENCE_SFP_RAIL_ORIGINS[pair.rail]
        placed_origin = SFP_RAIL_ORIGINS[pair.rail]
        sc_correction = _rotate_xyz(
            tuple(reference_origin[axis] - placed_origin[axis] for axis in range(3)),
            layout.board_rpy,
        )
        segment_lengths = tuple(
            sum((end[axis] - start[axis]) ** 2 for axis in range(3)) ** 0.5
            for start, end in zip(centerline, centerline[1:], strict=False)
        )
        cumulative_lengths = [0.0]
        for segment_length in segment_lengths:
            cumulative_lengths.append(cumulative_lengths[-1] + segment_length)
        centerline = tuple(
            tuple(
                point[axis] + sc_correction[axis] * cumulative_lengths[index] / cumulative_lengths[-1]
                for axis in range(3)
            )
            for index, point in enumerate(centerline)
        )
        sc_xyz = tuple(sc_xyz[axis] + sc_correction[axis] for axis in range(3))
        centerline = _add_endpoint_strain_reliefs(
            centerline,
            lc_plug_quat_xyzw=lc_quat,
            sc_plug_quat_xyzw=sc_quat,
            length=CABLE.endpoint_straight_length_m,
            blend_length=CABLE.endpoint_blend_length_m,
            max_segment_length=reference.max_segment_length_m,
        )
        assemblies.append(
            StaticCableAssembly(
                name=f"cable_{pair.index}",
                centerline=centerline,
                cable_xyz=cable_xyz,
                lc_plug_xyz=lc_xyz,
                lc_plug_quat_xyzw=lc_quat,
                sfp_module_xyz=sfp_xyz,
                sfp_module_quat_xyzw=sfp_quat,
                sc_plug_xyz=sc_xyz,
                sc_plug_quat_xyzw=sc_quat,
            )
        )
    return tuple(assemblies)


def manipulation_frames(
    layout: SceneLayout,
    assembly: StaticCableAssembly,
    nic_sdf_path: Path,
    *,
    nic_card_index: int = 0,
    nic_port_index: int = 0,
) -> ManipulationFrames:
    """Compose reviewed SFP mount and NIC port frames for one cable."""
    try:
        root = ElementTree.parse(nic_sdf_path).getroot()
    except (ElementTree.ParseError, OSError) as error:
        raise ValueError(f"Unable to read NIC SDF at {nic_sdf_path}") from error

    port_local = _sdf_link_pose(root, f"sfp_port_{nic_port_index}_link")
    entrance_local = _sdf_link_pose(root, f"sfp_port_{nic_port_index}_link_entrance")
    placements = {placement.name: placement for placement in component_placements(layout)}
    try:
        nic_card = placements[f"nic_card_{nic_card_index}"]
    except KeyError as error:
        raise ValueError(f"AIC layout does not define placement '{error.args[0]}'") from error

    nic_card_pose = _pose_tuple(
        nic_card.xyz,
        _rpy_quaternion(nic_card.rpy),
        field=nic_card.name,
    )
    raw_port_bottom_xyz, raw_port_bottom_quat = _compose_pose(
        nic_card_pose.xyz,
        nic_card_pose.quat_xyzw,
        port_local.xyz,
        port_local.quat_xyzw,
    )
    raw_port_bottom = _pose_tuple(
        raw_port_bottom_xyz,
        raw_port_bottom_quat,
        field="raw_port_bottom",
    )
    raw_port_entrance_xyz, raw_port_entrance_quat = _compose_pose(
        raw_port_bottom.xyz,
        raw_port_bottom.quat_xyzw,
        entrance_local.xyz,
        entrance_local.quat_xyzw,
    )
    raw_port_entrance = _pose_tuple(
        raw_port_entrance_xyz,
        raw_port_entrance_quat,
        field="raw_port_entrance",
    )
    port_bottom = _sfp_target_pose_at_port(raw_port_bottom)
    port_entrance = _sfp_target_pose_at_port(raw_port_entrance)
    board_quat = _normalize_quaternion(_rpy_quaternion(layout.board_rpy), field="board.quat_xyzw")
    board_normal = _normalize_direction(_quat_rotate(board_quat, (0.0, 0.0, 1.0)), field="board_normal")

    return ManipulationFrames(
        cable_name=assembly.name,
        sfp_module=_pose_tuple(
            assembly.sfp_module_xyz,
            assembly.sfp_module_quat_xyzw,
            field=f"{assembly.name}.sfp_module",
        ),
        port_bottom=port_bottom,
        port_entrance=port_entrance,
        board_normal=board_normal,
        mount_extraction_axis=board_normal,
    )


def _sfp_target_pose_at_port(port_pose: PoseTuple) -> PoseTuple:
    """Place the SFP mating face at an AIC NIC port frame."""
    _, sfp_quat = _compose_pose(
        port_pose.xyz,
        port_pose.quat_xyzw,
        (0.0, 0.0, 0.0),
        _rpy_quaternion(SFP_TARGET_FROM_RAW_PORT_RPY),
    )
    tip_offset_world = _quat_rotate(sfp_quat, (0.0, -SFP_FACE_OFFSET_M, 0.0))
    sfp_xyz = tuple(port_pose.xyz[axis] - tip_offset_world[axis] for axis in range(3))
    return _pose_tuple(sfp_xyz, sfp_quat, field="sfp_port_target")


def _rotate_xyz(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Rotate a point using fixed-axis roll, pitch, and yaw."""
    x, y, z = xyz
    roll, pitch, yaw = rpy
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    return (
        (cy * cp) * x + (cy * sp * sr - sy * cr) * y + (cy * sp * cr + sy * sr) * z,
        (sy * cp) * x + (sy * sp * sr + cy * cr) * y + (sy * sp * cr - cy * sr) * z,
        (-sp) * x + (cp * sr) * y + (cp * cr) * z,
    )


def _world_xyz(layout: SceneLayout, local_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    """Transform a task-board-local point into world coordinates."""
    rotated = _rotate_xyz(local_xyz, layout.board_rpy)
    return tuple(origin + offset for origin, offset in zip(layout.board_xyz, rotated, strict=True))


def component_placements(layout: SceneLayout) -> tuple[ComponentPlacement, ...]:
    """Return all indexed AIC component placements."""
    placements: list[ComponentPlacement] = []
    board_rpy = layout.board_rpy

    for pair in layout.cable_pairs:
        sfp_origin = SFP_RAIL_ORIGINS[pair.rail]
        sc_origin = SC_RAIL_ORIGINS[pair.rail]
        sfp_xyz = _world_xyz(layout, (sfp_origin[0], sfp_origin[1] + pair.translation, sfp_origin[2]))
        sc_xyz = _world_xyz(layout, (sc_origin[0], sc_origin[1] + pair.translation, sc_origin[2]))
        placements.extend(
            (
                ComponentPlacement(f"sfp_mount_{pair.index}", "SFP Mount", sfp_xyz, board_rpy),
                ComponentPlacement(f"sc_mount_{pair.index}", "SC Mount", sc_xyz, board_rpy),
                ComponentPlacement(f"cable_{pair.index}", "Cable", sfp_xyz, board_rpy),
                ComponentPlacement(f"sfp_module_{pair.index}", "SFP Module", sfp_xyz, board_rpy),
                ComponentPlacement(f"sc_plug_{pair.index}", "SC Plug", sc_xyz, board_rpy),
            )
        )

    for index, translation in enumerate(layout.nic_translations):
        mount_xyz = _world_xyz(layout, (-0.081418 + translation, NIC_RAIL_Y[index], 0.012))
        mount_quat = _rpy_quaternion(board_rpy)
        card_xyz, card_quat = _compose_pose(
            mount_xyz,
            mount_quat,
            (-0.002, -0.01785, 0.0899),
            _rpy_quaternion((-1.57, 0.0, 0.0)),
        )
        placements.extend(
            (
                ComponentPlacement(f"nic_card_mount_{index}", "NIC Card Mount", mount_xyz, board_rpy),
                ComponentPlacement(
                    f"nic_card_{index}",
                    "NIC Card",
                    card_xyz,
                    _quaternion_rpy(card_quat),
                ),
            )
        )

    for index, translation in enumerate(layout.sc_port_translations):
        port_xyz = _world_xyz(layout, (-0.075 + translation, SC_PORT_ROW_Y[index], 0.0165))
        _, port_quat = _compose_pose(
            port_xyz,
            _rpy_quaternion(board_rpy),
            (0.0, 0.0, 0.0),
            _rpy_quaternion((1.57, 0.0, 1.57)),
        )
        port_rpy = _quaternion_rpy(port_quat)
        placements.append(ComponentPlacement(f"sc_port_{index}", "SC Port", port_xyz, port_rpy))

    return tuple(placements)
