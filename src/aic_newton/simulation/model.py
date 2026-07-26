"""Build the shared coupled simulation model."""

from dataclasses import dataclass
from pathlib import Path

import newton
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupledProxy

from ..config import SIMULATION, SOLVER, TASK_TARGET, TaskTargetConfig
from ..scene.assets import mjcf_dir, scene_asset_dir, visual_model_dir
from ..scene.builder import (
    _add_aic_visuals,
    _add_task_board,
    _add_ur5_visuals,
    _load_static_scene_poses,
    _world_without_task_board,
)
from ..scene.cable import _world_without_original_cable, load_cable_reference
from ..scene.cable_builder import _add_dynamic_cable
from ..scene.collisions import (
    _filter_fixed_grasp_collisions,
    _filter_intersecting_sfp_board_shape_pairs,
)
from ..scene.layout import ManipulationFrames, SceneHandles, default_layout, manipulation_frames
from ..scene.robot import (
    _configure_mujoco_gravity_compensation,
    _set_robot_home,
)
from ..scene.visuals import _hide_shape_range
from ..utils.labels import find_label_index
from ..utils.transforms import transform_from_pose
from .attachments import DynamicCableHandles
from .solver import _build_coupled_solver


@dataclass(frozen=True)
class SimulationModel:
    """Group the shared model, solver, and indexed scene handles."""

    model: newton.Model
    solver: SolverCoupledProxy
    scene: SceneHandles
    cable: DynamicCableHandles
    manipulation_frames: ManipulationFrames
    robot_path: Path
    tool_body: int
    insertion_collision_shapes: tuple[int, ...]


def build_simulation_model(task_target: TaskTargetConfig = TASK_TARGET) -> SimulationModel:
    """Build the single physical scene used by all control modes."""
    model_dir = mjcf_dir()
    asset_dir = visual_model_dir()
    robot_path = model_dir / "aic_robot.xml"
    world_path = model_dir / "aic_world.xml"
    robot_source = robot_path.read_text(encoding="utf-8")
    world_source = world_path.read_text(encoding="utf-8")
    scene_poses = _load_static_scene_poses(world_path)
    layout = default_layout()
    reference = load_cable_reference(scene_asset_dir() / "cable_reference.json")

    builder = newton.ModelBuilder(gravity=SIMULATION.gravity_m_s2)
    builder.rigid_contact_max = SIMULATION.rigid_contact_max
    builder.rigid_gap = SIMULATION.rigid_gap_m
    SolverMuJoCo.register_custom_attributes(builder)
    SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)

    world_shape_start = builder.shape_count
    builder.add_mjcf(
        _world_without_task_board(_world_without_original_cable(world_source)),
        parse_meshes=False,
        parse_visuals=False,
        parse_sites=False,
        collapse_fixed_joints=True,
        skip_equality_constraints=True,
    )
    _hide_shape_range(builder, world_shape_start)

    robot_body_start = builder.body_count
    robot_joint_start = builder.joint_count
    robot_shape_start = builder.shape_count
    builder.add_mjcf(
        str(robot_path),
        parse_meshes=False,
        parse_visuals=False,
        parse_sites=False,
        enable_self_collisions=False,
        skip_equality_constraints=True,
    )
    _hide_shape_range(builder, robot_shape_start)
    _set_robot_home(builder)
    robot_bodies = list(range(robot_body_start, builder.body_count))
    robot_joints = list(range(robot_joint_start, builder.joint_count))
    tool_body = find_label_index(builder.body_label, "ati/tool_link")
    finger_bodies = (
        find_label_index(builder.body_label, "gripper/hande_finger_link_l"),
        find_label_index(builder.body_label, "gripper/hande_finger_link_r"),
    )
    _configure_mujoco_gravity_compensation(builder, robot_bodies)

    scene = _add_task_board(
        builder,
        layout,
        reference,
        asset_dir,
        skip_cable_names=frozenset({task_target.cable_name}),
    )
    try:
        target_assembly = scene.cable_assemblies[task_target.cable_index]
    except IndexError as error:
        raise ValueError("Configured task target index is outside the scene layout") from error
    frames = manipulation_frames(
        layout,
        target_assembly,
        asset_dir / "NIC Card" / "model.sdf",
        nic_card_index=task_target.nic_card_index,
        nic_port_index=task_target.nic_port_index,
    )
    cable = _add_dynamic_cable(
        builder,
        target_assembly,
        asset_dir,
        tool_body=tool_body,
        seat_module_pose=transform_from_pose(frames.port_bottom),
    )
    _filter_fixed_grasp_collisions(
        builder,
        connector_bodies=(cable.sfp_body, cable.sc_body),
        gripper_bodies=(tool_body, *finger_bodies),
    )
    _filter_intersecting_sfp_board_shape_pairs(
        builder,
        sfp_body=cable.sfp_body,
        board_shapes=scene.base_shapes,
    )
    insertion_bodies = {cable.sfp_body, tool_body, *finger_bodies}
    insertion_collision_shapes = tuple(
        shape
        for shape, body in enumerate(builder.shape_body)
        if body in insertion_bodies and builder.shape_flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES)
    )

    builder.color()
    _add_ur5_visuals(builder, robot_source)
    _add_aic_visuals(
        builder,
        asset_dir=asset_dir,
        robot_source=robot_source,
        scene_poses=scene_poses,
    )
    model = builder.finalize()
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    solver = _build_coupled_solver(
        model=model,
        robot_bodies=robot_bodies,
        robot_joints=robot_joints,
        proxy_bodies=[tool_body, *finger_bodies],
        payload_bodies=list(cable.bodies),
        payload_joints=list(cable.joints),
        vbd_iterations=SOLVER.vbd_iterations,
    )
    return SimulationModel(
        model=model,
        solver=solver,
        scene=scene,
        cable=cable,
        manipulation_frames=frames,
        robot_path=robot_path,
        tool_body=tool_body,
        insertion_collision_shapes=insertion_collision_shapes,
    )
