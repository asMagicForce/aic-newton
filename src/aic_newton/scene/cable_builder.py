"""Build static and dynamic Newton cable models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import newton
import numpy as np
import warp as wp

from ..config import CABLE
from ..simulation.attachments import DynamicCableHandles, _add_vbd_grasp_joint
from ..utils.transforms import transform_from_components
from .cable import _attachment_filter_segment_count
from .collisions import (
    _disable_aggregate_removed_connector_collisions,
    _filter_body_pair,
    _filter_grasped_cable_region,
)
from .visuals import _add_glb_visual, _add_sdf_box_colliders

if TYPE_CHECKING:
    from .cable import CableReference
    from .layout import SceneLayout, StaticCableAssembly


def _add_static_cable_assembly(
    builder: newton.ModelBuilder,
    assembly: StaticCableAssembly,
    *,
    asset_dir: Path,
    cache: dict[Path, newton.ModelBuilder],
) -> list[int]:
    """Add one static AIC cable and its connector visuals."""
    shapes: list[int] = []
    positions = [wp.vec3(*point) for point in assembly.centerline]
    quaternions = newton.utils.create_parallel_transport_cable_quaternions(positions)
    cable_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=CABLE.contact_stiffness_n_m,
        kd=CABLE.contact_damping_n_s_m,
        mu=CABLE.friction_coefficient,
        margin=CABLE.contact_margin_m,
        gap=CABLE.contact_gap_m,
        has_shape_collision=True,
        has_particle_collision=False,
        is_visible=True,
    )
    for index, (start, end, quat) in enumerate(zip(positions[:-1], positions[1:], quaternions, strict=True)):
        segment = end - start
        shapes.append(
            builder.add_shape_capsule(
                -1,
                xform=wp.transform(0.5 * (start + end), quat),
                radius=CABLE.radius_m,
                half_height=0.5 * float(wp.length(segment)),
                cfg=cable_cfg,
                color=wp.vec3(*CABLE.color_rgb),
                label=f"{assembly.name}/segment_{index}",
            )
        )

    for relative_path, xyz, quaternion, label in (
        (
            "LC Plug/lc_plug_visual.glb",
            assembly.lc_plug_xyz,
            assembly.lc_plug_quat_xyzw,
            f"{assembly.name}/lc_plug",
        ),
        (
            "SFP Module/sfp_module_visual.glb",
            assembly.sfp_module_xyz,
            assembly.sfp_module_quat_xyzw,
            f"{assembly.name}/sfp_module",
        ),
        (
            "SC Plug/sc_plug_visual.glb",
            assembly.sc_plug_xyz,
            assembly.sc_plug_quat_xyzw,
            f"{assembly.name}/sc_plug",
        ),
    ):
        shapes.extend(
            _add_glb_visual(
                builder,
                asset_dir=asset_dir,
                relative_path=relative_path,
                body=-1,
                xform=transform_from_components(xyz, quaternion),
                cache=cache,
                label=label,
            )
        )

    collider_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=True,
        has_particle_collision=False,
        is_visible=False,
    )
    shapes.append(
        builder.add_shape_box(
            -1,
            xform=transform_from_components(assembly.lc_plug_xyz, assembly.lc_plug_quat_xyzw),
            hx=0.0062,
            hy=0.012,
            hz=0.0031,
            cfg=collider_cfg,
            label=f"{assembly.name}/lc_plug_collision",
        )
    )
    shapes.append(
        builder.add_shape_box(
            -1,
            xform=transform_from_components(assembly.sc_plug_xyz, assembly.sc_plug_quat_xyzw),
            hx=0.022,
            hy=0.0125,
            hz=0.0051,
            cfg=collider_cfg,
            label=f"{assembly.name}/sc_plug_collision",
        )
    )
    return shapes


def _add_static_cables(
    builder: newton.ModelBuilder,
    layout: SceneLayout,
    reference: CableReference,
    asset_dir: Path,
    skip_names: frozenset[str] = frozenset(),
) -> tuple[tuple[StaticCableAssembly, ...], dict[str, list[int]]]:
    """Add the requested static AIC cable assemblies."""
    from .layout import build_static_cable_assemblies

    assemblies = build_static_cable_assemblies(layout, reference)
    cache: dict[Path, newton.ModelBuilder] = {}
    shapes = {
        assembly.name: _add_static_cable_assembly(
            builder,
            assembly,
            asset_dir=asset_dir,
            cache=cache,
        )
        for assembly in assemblies
        if assembly.name not in skip_names
    }
    return assemblies, shapes


def _add_rooted_rod(
    builder: newton.ModelBuilder,
    *,
    positions: list[wp.vec3],
    quaternions: list[wp.quat],
    radius: float,
    cfg: newton.ModelBuilder.ShapeConfig,
    stretch_stiffness: float = 1.0e5,
    stretch_damping: float = 0.0,
    bend_stiffness: float = 0.0,
    bend_damping: float = 0.0,
    label: str = "rod",
    color: wp.vec3 | None = None,
    root_parent: int | None = None,
    add_articulation: bool = True,
) -> tuple[list[int], list[int]]:
    """Add a free- or body-rooted VBD rod with topologically ordered joints."""
    segment_count = len(positions) - 1
    if segment_count < 2:
        raise ValueError("A rooted rod requires at least two segments")
    if len(quaternions) != segment_count:
        raise ValueError(f"Expected {segment_count} rod quaternions, got {len(quaternions)}")

    bodies: list[int] = []
    half_lengths: list[float] = []
    rod_color = color if color is not None else wp.vec3(0.06, 0.10, 0.16)
    for index, (start, end, quat) in enumerate(zip(positions[:-1], positions[1:], quaternions, strict=True)):
        segment = end - start
        length = float(wp.length(segment))
        if length <= 1.0e-9:
            raise ValueError(f"Rod segment {index} has zero length")
        half_length = 0.5 * length
        body = builder.add_link(
            xform=wp.transform(start + segment * 0.5, quat),
            label=f"{label}_body_{index}",
        )
        builder.add_shape_capsule(
            body,
            radius=radius,
            half_height=half_length,
            cfg=cfg,
            label=f"{label}_capsule_{index}",
            color=rod_color,
        )
        bodies.append(body)
        half_lengths.append(half_length)

    if root_parent is None:
        joints = [builder.add_joint_free(child=bodies[0], label=f"{label}_root")]
    else:
        endpoint_frame = wp.transform(positions[0], quaternions[0])
        joints = [
            builder.add_joint_fixed(
                parent=root_parent,
                child=bodies[0],
                parent_xform=wp.transform_inverse(builder.body_q[root_parent]) * endpoint_frame,
                child_xform=wp.transform(
                    wp.vec3(0.0, 0.0, -half_lengths[0]),
                    wp.quat_identity(),
                ),
                collision_filter_parent=True,
                label=f"{label}_root_attachment",
            )
        ]
    for index in range(1, segment_count):
        joints.append(
            builder.add_joint_cable(
                parent=bodies[index - 1],
                child=bodies[index],
                parent_xform=wp.transform(
                    wp.vec3(0.0, 0.0, half_lengths[index - 1]),
                    wp.quat_identity(),
                ),
                child_xform=wp.transform(
                    wp.vec3(0.0, 0.0, -half_lengths[index]),
                    wp.quat_identity(),
                ),
                stretch_stiffness=stretch_stiffness,
                stretch_damping=stretch_damping,
                bend_stiffness=bend_stiffness,
                bend_damping=bend_damping,
                collision_filter_parent=True,
                label=f"{label}_joint_{index - 1}_{index}",
            )
        )
    if add_articulation:
        builder.add_articulation(joints, label=label)
    return bodies, joints


def _add_dynamic_cable(
    builder: newton.ModelBuilder,
    assembly: StaticCableAssembly,
    asset_dir: Path,
    *,
    tool_body: int,
    seat_module_pose: wp.transform | None = None,
) -> DynamicCableHandles:
    """Build cable 0 as one dynamic connector-and-rod articulation."""
    assembly_shape_start = builder.shape_count
    cable_points = [wp.vec3(*point) for point in assembly.centerline]
    cable_half_lengths = tuple(
        0.5 * float(wp.length(end - start)) for start, end in zip(cable_points[:-1], cable_points[1:], strict=True)
    )
    cable_quaternions = newton.utils.create_parallel_transport_cable_quaternions(cable_points)
    sfp_pose = transform_from_components(assembly.lc_plug_xyz, assembly.lc_plug_quat_xyzw)
    sc_pose = transform_from_components(assembly.sc_plug_xyz, assembly.sc_plug_quat_xyzw)
    sfp_module_pose = transform_from_components(assembly.sfp_module_xyz, assembly.sfp_module_quat_xyzw)
    if seat_module_pose is None:
        seat_module_pose = sfp_module_pose
    sfp_body_to_module = wp.transform_inverse(sfp_pose) * sfp_module_pose
    seat_sfp_pose = seat_module_pose * wp.transform_inverse(sfp_body_to_module)

    sfp_body = builder.add_link(xform=sfp_pose, label=f"{assembly.name}/sfp_side")
    root_joint = builder.add_joint_free(child=sfp_body, enabled=False, label=f"{assembly.name}/sfp_root")
    cable_cfg = newton.ModelBuilder.ShapeConfig(
        density=CABLE.density_kg_m3,
        ke=CABLE.contact_stiffness_n_m,
        kd=CABLE.contact_damping_n_s_m,
        mu=CABLE.friction_coefficient,
        margin=CABLE.contact_margin_m,
        gap=CABLE.contact_gap_m,
        has_particle_collision=False,
    )
    cable_bodies, rooted_cable_joints = _add_rooted_rod(
        builder,
        positions=cable_points,
        quaternions=cable_quaternions,
        radius=CABLE.radius_m,
        cfg=cable_cfg,
        stretch_stiffness=CABLE.stretch_stiffness_n_m,
        stretch_damping=CABLE.stretch_damping_n_s_m,
        bend_stiffness=CABLE.bend_stiffness,
        bend_damping=CABLE.bend_damping,
        label=assembly.name,
        color=wp.vec3(*CABLE.color_rgb),
        root_parent=sfp_body,
        add_articulation=False,
    )

    sc_body = builder.add_link(xform=sc_pose, label=f"{assembly.name}/sc_plug")
    last_half_length = 0.5 * float(wp.length(cable_points[-1] - cable_points[-2]))
    endpoint_frame = wp.transform(cable_points[-1], cable_quaternions[-1])
    sc_attachment_joint = builder.add_joint_fixed(
        parent=cable_bodies[-1],
        child=sc_body,
        parent_xform=wp.transform(
            wp.vec3(0.0, 0.0, last_half_length),
            wp.quat_identity(),
        ),
        child_xform=wp.transform_inverse(sc_pose) * endpoint_frame,
        collision_filter_parent=True,
        label=f"{assembly.name}/sc_attachment",
    )
    joints = (root_joint, *rooted_cable_joints, sc_attachment_joint)
    builder.add_articulation(list(joints), label=assembly.name)
    mount_anchor_body = builder.add_link(
        xform=sfp_pose,
        mass=0.0,
        is_kinematic=True,
        label=f"{assembly.name}/mount_anchor",
    )
    seat_anchor_body = builder.add_link(
        xform=seat_sfp_pose,
        mass=0.0,
        is_kinematic=True,
        label=f"{assembly.name}/seat_anchor",
    )
    sc_anchor_body = builder.add_link(
        xform=sc_pose,
        mass=0.0,
        is_kinematic=True,
        label=f"{assembly.name}/sc_mount_anchor",
    )
    mount_joint = builder.add_joint_fixed(
        parent=mount_anchor_body,
        child=sfp_body,
        collision_filter_parent=False,
        label=f"{assembly.name}/mount_attachment",
    )
    grasp_joint = _add_vbd_grasp_joint(
        builder,
        tool_body=tool_body,
        sfp_body=sfp_body,
        label=f"{assembly.name}/fixed_grasp",
    )
    seat_joint = builder.add_joint_fixed(
        parent=seat_anchor_body,
        child=sfp_body,
        collision_filter_parent=False,
        enabled=False,
        label=f"{assembly.name}/seat_attachment",
    )
    sc_mount_joint = builder.add_joint_fixed(
        parent=sc_anchor_body,
        child=sc_body,
        collision_filter_parent=False,
        label=f"{assembly.name}/sc_mount_attachment",
    )
    joints = (*joints, mount_joint, grasp_joint, seat_joint, sc_mount_joint)

    connector_cfg = newton.ModelBuilder.ShapeConfig(
        density=CABLE.connector_density_kg_m3,
        ke=CABLE.contact_stiffness_n_m,
        kd=CABLE.contact_damping_n_s_m,
        mu=CABLE.friction_coefficient,
        margin=CABLE.connector_contact_margin_m,
        gap=CABLE.connector_contact_gap_m,
        has_shape_collision=True,
        has_particle_collision=False,
        is_visible=False,
    )
    connector_specs = (
        ("LC Plug", "lc_plug_link", sfp_body, wp.transform_identity(), f"{assembly.name}/lc_plug_collision"),
        (
            "SFP Module",
            "sfp_module_link",
            sfp_body,
            sfp_body_to_module,
            f"{assembly.name}/sfp_module_collision",
        ),
        (
            "SFP Module",
            "sfp_tip_link",
            sfp_body,
            sfp_body_to_module,
            f"{assembly.name}/sfp_tip_collision",
        ),
        ("SC Plug", "sc_plug_link", sc_body, wp.transform_identity(), f"{assembly.name}/sc_plug_collision"),
    )
    for asset_name, link_name, body, local_pose, label in connector_specs:
        collision_shapes = _add_sdf_box_colliders(
            builder,
            model_path=asset_dir / asset_name / "model.sdf",
            parent_pose=local_pose,
            link_name=link_name,
            label_prefix=label,
            body=body,
            cfg=connector_cfg,
        )
        _disable_aggregate_removed_connector_collisions(
            builder,
            asset_name=asset_name,
            link_name=link_name,
            shapes=collision_shapes,
        )

    assembly_bodies = frozenset((sfp_body, *cable_bodies, sc_body))
    assembly_collision_shapes = [
        shape
        for shape in range(assembly_shape_start, builder.shape_count)
        if builder.shape_body[shape] in assembly_bodies
        and builder.shape_flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES)
    ]
    for index, shape_a in enumerate(assembly_collision_shapes):
        for shape_b in assembly_collision_shapes[index + 1 :]:
            builder.add_shape_collision_filter_pair(shape_a, shape_b)

    cache: dict[Path, newton.ModelBuilder] = {}
    for relative_path, body, local_pose, label in (
        ("LC Plug/lc_plug_visual.glb", sfp_body, wp.transform_identity(), f"{assembly.name}/lc_plug"),
        (
            "SFP Module/sfp_module_visual.glb",
            sfp_body,
            sfp_body_to_module,
            f"{assembly.name}/sfp_module",
        ),
        ("SC Plug/sc_plug_visual.glb", sc_body, wp.transform_identity(), f"{assembly.name}/sc_plug"),
    ):
        _add_glb_visual(
            builder,
            asset_dir=asset_dir,
            relative_path=relative_path,
            body=body,
            xform=local_pose,
            cache=cache,
            label=label,
        )

    attachment_filter_count = _attachment_filter_segment_count(
        np.asarray(cable_points),
        CABLE.attachment_filter_length_m,
    )
    _filter_grasped_cable_region(
        builder,
        tool_body=tool_body,
        plug_body=sfp_body,
        cable_bodies=cable_bodies[:attachment_filter_count],
    )
    for cable_body in cable_bodies[-attachment_filter_count:]:
        _filter_body_pair(builder, sc_body, cable_body)

    return DynamicCableHandles(
        sfp_body=sfp_body,
        sc_body=sc_body,
        cable_bodies=tuple(cable_bodies),
        cable_half_lengths=cable_half_lengths,
        sfp_root_joint=root_joint,
        mount_anchor_body=mount_anchor_body,
        seat_anchor_body=seat_anchor_body,
        sc_anchor_body=sc_anchor_body,
        mount_joint=mount_joint,
        grasp_joint=grasp_joint,
        seat_joint=seat_joint,
        sc_mount_joint=sc_mount_joint,
        bodies=(
            sfp_body,
            *cable_bodies,
            sc_body,
            mount_anchor_body,
            seat_anchor_body,
            sc_anchor_body,
        ),
        joints=joints,
        sfp_body_to_module=sfp_body_to_module,
    )
