"""Build the static workcell and its render-only geometry."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import newton
import warp as wp

from .cable import CableReference, _body_poses_by_suffix
from .cable_builder import _add_static_cables
from .layout import SceneHandles, SceneLayout, component_placements
from .visuals import _add_glb_visual, _add_sdf_box_colliders


def _world_without_task_board(source: str) -> str:
    """Remove the source task-board subtree and its contacts."""
    root = ET.fromstring(source)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("AIC world MJCF has no <worldbody> element")

    removed_names: set[str] = set()
    for body in list(worldbody):
        if body.tag != "body" or body.get("name") != "task_board_base_link":
            continue
        removed_names.update(child.get("name") for child in body.iter("body") if child.get("name") is not None)
        worldbody.remove(body)
        break
    if "task_board_base_link" not in removed_names:
        raise ValueError("AIC world MJCF has no task_board_base_link body")

    contact = root.find("contact")
    if contact is not None:
        for exclude in list(contact):
            if exclude.get("body1") in removed_names or exclude.get("body2") in removed_names:
                contact.remove(exclude)
        if not list(contact):
            root.remove(contact)
    return ET.tostring(root, encoding="unicode")


def _load_static_scene_poses(world_path: Path) -> dict[str, wp.transform]:
    """Load enclosure and floor poses from the vendored world."""
    builder = newton.ModelBuilder()
    builder.add_mjcf(
        str(world_path),
        parse_meshes=False,
        parse_visuals=False,
        parse_sites=False,
        skip_equality_constraints=True,
    )
    return _body_poses_by_suffix(
        builder.body_label,
        builder.body_q,
        ("enclosure_link", "floor_link"),
    )


def _add_task_board(
    builder: newton.ModelBuilder,
    layout: SceneLayout,
    reference: CableReference,
    asset_dir: Path,
    *,
    skip_cable_names: frozenset[str] = frozenset(),
) -> SceneHandles:
    """Add the complete static AIC task-board scene."""
    cache: dict[Path, newton.ModelBuilder] = {}
    board_pose = wp.transform(wp.vec3(*layout.board_xyz), wp.quat_rpy(*layout.board_rpy))
    base_shapes = _add_glb_visual(
        builder,
        asset_dir=asset_dir,
        relative_path="Task Board Base/base_visual.glb",
        body=-1,
        xform=board_pose,
        cache=cache,
        label="scene/task_board",
    )
    base_shapes.extend(
        _add_sdf_box_colliders(
            builder,
            model_path=asset_dir / "Task Board Base" / "model.sdf",
            parent_pose=board_pose,
            link_name="base_link",
            label_prefix="scene/task_board",
        )
    )

    placements = component_placements(layout)
    by_asset = {
        asset_name: [placement for placement in placements if placement.asset_name == asset_name]
        for asset_name in ("SFP Mount", "SC Mount", "SC Port", "NIC Card Mount", "NIC Card")
    }
    asset_specs = {
        "SFP Mount": ("SFP Mount/sfp_mount_visual.glb", "sfp_mount_link"),
        "SC Mount": ("SC Mount/sc_mount_visual.glb", "sc_mount_link"),
        "SC Port": ("SC Port/sc_port_visual.glb", "sc_port_link"),
        "NIC Card Mount": ("NIC Card Mount/nic_card_mount_visual.glb", "nic_card_mount_link"),
        "NIC Card": ("NIC Card/nic_card_visual.glb", "nic_card_link"),
    }

    component_shapes: dict[str, tuple[tuple[int, ...], ...]] = {}
    for asset_name, items in by_asset.items():
        relative_path, link_name = asset_specs[asset_name]
        collider_cfg = None
        if asset_name == "SFP Mount":
            collider_cfg = newton.ModelBuilder.ShapeConfig(
                density=0.0,
                gap=0.0,
                has_shape_collision=True,
                has_particle_collision=False,
                is_visible=False,
            )
        groups: list[tuple[int, ...]] = []
        for placement in items:
            pose = wp.transform(wp.vec3(*placement.xyz), wp.quat_rpy(*placement.rpy))
            shapes = _add_glb_visual(
                builder,
                asset_dir=asset_dir,
                relative_path=relative_path,
                body=-1,
                xform=pose,
                cache=cache,
                label=f"scene/{placement.name}",
            )
            shapes.extend(
                _add_sdf_box_colliders(
                    builder,
                    model_path=asset_dir / asset_name / "model.sdf",
                    parent_pose=pose,
                    link_name=link_name,
                    label_prefix=f"scene/{placement.name}",
                    cfg=collider_cfg,
                )
            )
            groups.append(tuple(shapes))
        component_shapes[asset_name] = tuple(groups)

    cable_assemblies, cable_shapes = _add_static_cables(
        builder,
        layout,
        reference,
        asset_dir,
        skip_names=skip_cable_names,
    )
    return SceneHandles(
        base_shapes=tuple(base_shapes),
        sfp_mount_shapes=component_shapes["SFP Mount"],
        sc_port_shapes=component_shapes["SC Port"],
        nic_card_shapes=component_shapes["NIC Card"],
        cable_assemblies=cable_assemblies,
        cable_shapes=cable_shapes,
    )


def _add_environment_visuals(
    builder: newton.ModelBuilder,
    *,
    asset_dir: Path,
    scene_poses: dict[str, wp.transform],
) -> None:
    """Attach environment visuals not included in the robot description."""
    cache: dict[Path, newton.ModelBuilder] = {}
    copied: list[int] = []

    def add(
        relative_path: str,
        body: int,
        xform: wp.transform,
        label: str,
    ) -> None:
        copied.extend(
            _add_glb_visual(
                builder,
                asset_dir=asset_dir,
                relative_path=relative_path,
                body=body,
                xform=xform,
                cache=cache,
                label=label,
            )
        )

    enclosure_pose = scene_poses["enclosure_link"]
    add("Enclosure/enclosure_visual.glb", -1, enclosure_pose, "enclosure")
    add("Enclosure/light_visual.glb", -1, enclosure_pose, "enclosure_light")
    floor_pose = scene_poses["floor_link"]
    add("Floor/floor_visual.glb", -1, floor_pose, "floor")
    add("Floor/walls_visual.glb", -1, floor_pose, "floor_walls")
