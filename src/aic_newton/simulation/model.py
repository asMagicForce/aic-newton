"""Build the shared coupled simulation model."""

from dataclasses import dataclass
from math import pi
from pathlib import Path

import newton
import warp as wp
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupledProxy

from ..config import SIMULATION, SOLVER, TASK_TARGET, TaskTargetConfig
from ..scene.assets import mjcf_dir, robot_description_path, scene_asset_dir, visual_model_dir
from ..scene.builder import (
    _add_environment_visuals,
    _add_task_board,
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
    AIC_ARM_JOINT_NAMES,
    AIC_ROBOT_MODEL,
    _configure_mujoco_gravity_compensation,
    _set_robot_home,
)
from ..scene.visuals import _hide_shape_range
from ..utils.labels import find_label_index
from ..utils.transforms import PoseTuple, rpy_quaternion, transform_from_pose
from .attachments import DynamicCableHandles
from .solver import _build_coupled_solver

ROBOT_ROOT_XYZ_M = (-0.2, 0.2, 1.14)
ROBOT_ROOT_RPY_RAD = (0.0, 0.0, pi)
ROBOT_ROOT_POSE = PoseTuple(ROBOT_ROOT_XYZ_M, rpy_quaternion(ROBOT_ROOT_RPY_RAD))
ROBOT_ROOT_XFORM = transform_from_pose(ROBOT_ROOT_POSE)
# The URDF's fixed accessory chain changes the effective mass seen by the
# lagged VBD proxy. These source-body values reproduce the validated UR5e
# proxy (7.18 kg and 0.2768 kg m^2) after coupling, without changing geometry.
GRASP_PROXY_MASS_KG = 1.93864
GRASP_PROXY_INERTIA_KG_M2 = 0.168788


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


@dataclass(frozen=True)
class RobotModel:
    """Observable metadata from Newton's ROS-aligned robot import."""

    model_name: str
    arm_joint_names: tuple[str, ...]
    tool_body_label: str
    camera_body_labels: tuple[str, ...]
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]


def _add_robot_description(
    builder: newton.ModelBuilder,
    *,
    path: Path,
    xform=None,
    hide_visuals: bool = True,
) -> None:
    builder.add_urdf(
        str(path),
        xform=xform,
        hide_visuals=hide_visuals,
        enable_self_collisions=False,
        collapse_fixed_joints=False,
        force_position_velocity_actuation=True,
    )


def _configure_grasp_proxy(builder: newton.ModelBuilder, body: int) -> None:
    """Preserve the validated grasp proxy's effective coupled dynamics."""
    builder.body_mass[body] = GRASP_PROXY_MASS_KG
    builder.body_com[body] = wp.vec3(0.0, 0.0, 0.0)
    builder.body_inertia[body] = wp.mat33(
        GRASP_PROXY_INERTIA_KG_M2,
        0.0,
        0.0,
        0.0,
        GRASP_PROXY_INERTIA_KG_M2,
        0.0,
        0.0,
        0.0,
        GRASP_PROXY_INERTIA_KG_M2,
    )


def build_robot_model(path: Path | None = None) -> RobotModel:
    """Import the workcell robot and return its canonical runtime metadata."""
    builder = newton.ModelBuilder(gravity=SIMULATION.gravity_m_s2)
    _add_robot_description(builder, path=path or robot_description_path())
    joint_indices = tuple(find_label_index(builder.joint_label, name) for name in AIC_ARM_JOINT_NAMES)
    dof_indices = tuple(int(builder.joint_qd_start[index]) for index in joint_indices)
    tool_body = find_label_index(builder.body_label, "hande_base_link")
    camera_bodies = tuple(
        find_label_index(builder.body_label, f"camera_{view}_optical") for view in ("left", "center", "right")
    )
    return RobotModel(
        model_name=AIC_ROBOT_MODEL,
        arm_joint_names=AIC_ARM_JOINT_NAMES,
        tool_body_label=builder.body_label[tool_body],
        camera_body_labels=tuple(builder.body_label[index] for index in camera_bodies),
        joint_lower=tuple(float(builder.joint_limit_lower[index]) for index in dof_indices),
        joint_upper=tuple(float(builder.joint_limit_upper[index]) for index in dof_indices),
    )


def build_simulation_model(task_target: TaskTargetConfig = TASK_TARGET) -> SimulationModel:
    """Build the single physical scene used by all control modes."""
    model_dir = mjcf_dir()
    asset_dir = visual_model_dir()
    robot_path = robot_description_path()
    world_path = model_dir / "aic_world.xml"
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
    _add_robot_description(
        builder,
        path=robot_path,
        xform=ROBOT_ROOT_XFORM,
        hide_visuals=False,
    )
    _set_robot_home(builder)
    robot_bodies = list(range(robot_body_start, builder.body_count))
    robot_joints = list(range(robot_joint_start, builder.joint_count))
    for shape, body in enumerate(builder.shape_body):
        if body in robot_bodies:
            builder.shape_flags[shape] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
    tool_body = find_label_index(builder.body_label, "hande_base_link")
    _configure_grasp_proxy(builder, tool_body)
    finger_bodies = (
        find_label_index(builder.body_label, "hande_left_finger"),
        find_label_index(builder.body_label, "hande_right_finger"),
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
    _add_environment_visuals(
        builder,
        asset_dir=asset_dir,
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
