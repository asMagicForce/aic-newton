"""Build the static workcell and its render-only geometry."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import newton
import warp as wp

from ..utils.labels import find_label_index as _find_label_index
from .cable import CableReference, _body_poses_by_suffix, _geom_transform
from .cable_builder import _add_static_cables
from .layout import SceneHandles, SceneLayout, component_placements
from .visuals import _add_glb_visual, _add_sdf_box_colliders, _copy_visual_mesh_shapes


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


def _ur5_visual_body_xforms(
    source: newton.ModelBuilder,
    robot_source: str,
) -> dict[int, wp.transform]:
    """Map structured UR5e mesh bodies through the generated MJCF frames."""
    visual_geoms = {
        "base": "tabletop_fixed_joint_lump__base_link_inertia_visual",
        "shoulder_link": "shoulder_link_visual",
        "upper_arm_link": "upper_arm_link_visual",
        "forearm_link": "forearm_link_visual",
        "wrist_1_link": "wrist_1_link_visual",
        "wrist_2_link": "wrist_2_link_visual",
        "wrist_3_link": "wrist_3_link_visual",
    }
    return {
        body: _geom_transform(robot_source, visual_geoms[label.rsplit("/", 1)[-1]])
        for body, label in enumerate(source.body_label)
        if label.rsplit("/", 1)[-1] in visual_geoms
    }


def _add_ur5_visuals(builder: newton.ModelBuilder, robot_source: str) -> None:
    """Attach Newton's structured UR5e visuals to the imported AIC robot."""
    asset_path = newton.utils.download_asset("universal_robots_ur5e")
    source = newton.ModelBuilder()
    source.add_usd(
        str(asset_path / "usd_structured" / "ur5e.usda"),
        hide_collision_shapes=True,
        enable_self_collisions=False,
    )
    target_suffixes = {
        "base": "tabletop",
        "shoulder_link": "shoulder_link",
        "upper_arm_link": "upper_arm_link",
        "forearm_link": "forearm_link",
        "wrist_1_link": "wrist_1_link",
        "wrist_2_link": "wrist_2_link",
        "wrist_3_link": "wrist_3_link",
    }
    body_map: dict[int, int] = {}
    for source_body, source_label in enumerate(source.body_label):
        source_name = source_label.rsplit("/", 1)[-1]
        target_suffix = target_suffixes.get(source_name)
        if target_suffix is not None:
            body_map[source_body] = _find_label_index(builder.body_label, target_suffix)

    copied = _copy_visual_mesh_shapes(
        source,
        builder,
        body_map=body_map,
        body_xforms=_ur5_visual_body_xforms(source, robot_source),
        label_prefix="ur5e",
    )
    if not copied:
        raise ValueError("Newton's UR5e asset contains no visible meshes")


def _add_aic_visuals(
    builder: newton.ModelBuilder,
    *,
    asset_dir: Path,
    robot_source: str,
    scene_poses: dict[str, wp.transform],
) -> None:
    """Attach AIC workcell, sensor, gripper, and robot visuals."""
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

    wrist = _find_label_index(builder.body_label, "wrist_3_link")
    tool = _find_label_index(builder.body_label, "ati/tool_link")
    left_finger = _find_label_index(builder.body_label, "gripper/hande_finger_link_l")
    right_finger = _find_label_index(builder.body_label, "gripper/hande_finger_link_r")
    add(
        "Camera Mount/cam_mount_visual.glb",
        wrist,
        _geom_transform(robot_source, "wrist_3_link_fixed_joint_lump__cam_mount_visual_visual_1"),
        "camera_mount",
    )
    add(
        "Axia80 M20/axia_ft_sensor_visual.glb",
        wrist,
        _geom_transform(robot_source, "wrist_3_link_fixed_joint_lump__axia_ft_sensor_visual_visual_3"),
        "force_torque_sensor",
    )
    for camera, geom_name in (
        ("center", "wrist_3_link_fixed_joint_lump__basler_cam_visual_visual_2"),
        ("left", "wrist_3_link_fixed_joint_lump__basler_cam_visual_visual_4"),
        ("right", "wrist_3_link_fixed_joint_lump__basler_cam_visual_visual_5"),
    ):
        add(
            "Basler Camera/basler_cam_visual.glb",
            wrist,
            _geom_transform(robot_source, geom_name),
            f"{camera}_camera",
        )
    add(
        "Robotiq Hand-E/hande_base_visual.glb",
        tool,
        _geom_transform(robot_source, "ati/tool_link_fixed_joint_lump__hande_base_visual_visual"),
        "gripper_base",
    )
    add(
        "Robotiq Hand-E/hande_finger_visual.glb",
        left_finger,
        _geom_transform(
            robot_source,
            "gripper/hande_finger_link_l_fixed_joint_lump__hande_finger_visual_visual",
        ),
        "left_finger",
    )
    add(
        "Robotiq Hand-E/hande_finger_visual.glb",
        right_finger,
        _geom_transform(
            robot_source,
            "gripper/hande_finger_link_r_fixed_joint_lump__hande_finger_visual_visual",
        ),
        "right_finger",
    )
