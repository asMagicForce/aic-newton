"""Maintain the Cartesian TCP target and solve robot inverse kinematics."""

from __future__ import annotations

from pathlib import Path

import newton
import newton.ik as ik
import warp as wp

from ..config import MANUAL_TCP, ROBOT_CONTROL
from ..scene.robot import (
    AIC_ARM_JOINT_NAMES,
    AIC_GRIPPER_TCP_OFFSET,
    AIC_TOOL_TO_GRIPPER_TCP,
    _arm_coordinate_indices,
    _set_robot_home,
)
from ..utils.labels import find_label_index
from ..utils.transforms import transform_from_row
from .manual_tcp import (
    integrate_tcp_target,
    limit_tcp_target_translation,
    read_tcp_keyboard,
    update_translation_stall,
)


@wp.kernel
def _write_indexed_joint_targets(
    current_q: wp.array2d[float],
    previous_q: wp.array2d[float],
    source_indices: wp.array[int],
    target_q: wp.array[float],
    target_qd: wp.array[float],
    destination_indices: wp.array[int],
    inverse_dt: float,
):
    index = wp.tid()
    source = source_indices[index]
    destination = destination_indices[index]
    position = current_q[0, source]
    target_q[destination] = position
    target_qd[destination] = (position - previous_q[0, source]) * inverse_dt
    previous_q[0, source] = position


class TCPController:
    """Solve a persistent Cartesian TCP target into arm joint targets."""

    def __init__(self, model: newton.Model, robot_path: Path):
        self.model = model
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            str(robot_path),
            parse_meshes=False,
            parse_visuals=False,
            parse_sites=False,
            enable_self_collisions=False,
            skip_equality_constraints=True,
        )
        _set_robot_home(builder)
        self.ik_model = builder.finalize(device=model.device)
        self.joint_q = wp.array(
            self.ik_model.joint_q,
            shape=(1, self.ik_model.joint_coord_count),
        )
        self.previous_joint_q = wp.clone(self.joint_q)

        state = self.ik_model.state()
        newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, state)
        tool = find_label_index(self.ik_model.body_label, "ati/tool_link")
        self.initial_target = transform_from_row(state.body_q.numpy()[tool]) * AIC_TOOL_TO_GRIPPER_TCP
        self.target = wp.transform(
            wp.transform_get_translation(self.initial_target),
            wp.transform_get_rotation(self.initial_target),
        )

        position = wp.transform_get_translation(self.target)
        rotation = wp.transform_get_rotation(self.target)
        self.position_objective = ik.IKObjectivePosition(
            link_index=tool,
            link_offset=AIC_GRIPPER_TCP_OFFSET,
            target_positions=wp.array([position], dtype=wp.vec3, device=model.device),
        )
        self.rotation_objective = ik.IKObjectiveRotation(
            link_index=tool,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array(
                [wp.vec4(rotation[0], rotation[1], rotation[2], rotation[3])],
                dtype=wp.vec4,
                device=model.device,
            ),
        )
        joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=ROBOT_CONTROL.ik_joint_limit_weight,
        )
        self.solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[self.position_objective, self.rotation_objective, joint_limits],
            lambda_initial=ROBOT_CONTROL.ik_damping_initial,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.iterations = ROBOT_CONTROL.ik_iterations
        self.ik_arm_coordinates = wp.array(
            _arm_coordinate_indices(self.ik_model),
            dtype=int,
            device=model.device,
        )
        self.control_arm_coordinates = wp.array(
            _arm_coordinate_indices(model),
            dtype=int,
            device=model.device,
        )
        self.control_arm_coordinate_indices = tuple(_arm_coordinate_indices(model))

        self._translation_stall_time = 0.0
        self._translation_blocked = False
        self._previous_position = wp.transform_get_translation(self.initial_target)

    def reset(self) -> None:
        """Restore the home IK state and clear manual tracking history."""
        self.joint_q.assign(self.ik_model.joint_q)
        self.previous_joint_q.assign(self.ik_model.joint_q)
        self.set_target(self.initial_target)
        self._translation_stall_time = 0.0
        self._translation_blocked = False
        self._previous_position = wp.transform_get_translation(self.initial_target)

    def set_target(self, target: wp.transform) -> None:
        """Write a Cartesian target into the existing IK objectives."""
        self.target = target
        position = wp.transform_get_translation(target)
        rotation = wp.transform_get_rotation(target)
        self.position_objective.set_target_position(0, position)
        self.rotation_objective.set_target_rotation(
            0,
            wp.vec4(rotation[0], rotation[1], rotation[2], rotation[3]),
        )

    def update_manual(self, viewer, state: newton.State, *, tool_body: int, dt: float) -> None:
        """Update the persistent target from viewer keyboard state."""
        linear_axis, angular_axis, _ = read_tcp_keyboard(viewer)
        tool_pose = transform_from_row(state.body_q.numpy()[tool_body])
        current_position = wp.transform_get_translation(tool_pose * AIC_TOOL_TO_GRIPPER_TCP)
        self._translation_stall_time, self._translation_blocked = update_translation_stall(
            linear_axis=linear_axis,
            current_position=current_position,
            previous_position=self._previous_position,
            target_position=wp.transform_get_translation(self.target),
            elapsed=self._translation_stall_time,
            blocked=self._translation_blocked,
            dt=dt,
        )
        if self._translation_blocked:
            self.target = wp.transform(
                current_position,
                wp.transform_get_rotation(self.target),
            )
            linear_axis = wp.vec3(0.0)
        self.target = integrate_tcp_target(
            self.target,
            linear_axis,
            angular_axis,
            dt,
        )
        self.target = limit_tcp_target_translation(
            self.target,
            current_position=current_position,
            max_error=MANUAL_TCP.max_tracking_error_m,
        )
        self._previous_position = current_position
        self.set_target(self.target)

    def step(self, control: newton.Control, *, dt: float) -> None:
        """Solve IK and copy the arm coordinates into simulation control."""
        self.solver.step(self.joint_q, self.joint_q, iterations=self.iterations)
        wp.launch(
            _write_indexed_joint_targets,
            dim=len(AIC_ARM_JOINT_NAMES),
            inputs=[
                self.joint_q,
                self.previous_joint_q,
                self.ik_arm_coordinates,
                control.joint_target_q,
                control.joint_target_qd,
                self.control_arm_coordinates,
                1.0 / dt,
            ],
            device=self.model.device,
        )
