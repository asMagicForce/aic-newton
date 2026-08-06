# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# AIC Task-Board Scene
#
# Imports the AIC robot and static workcell from vendored assets. The UR5e is
# simulated by SolverMuJoCo, while five initially docked cable assemblies and
# the AIC task-board components form a static Newton environment.
#
# Command: uv run aic
#
###########################################################################

from __future__ import annotations

from collections.abc import Callable

import newton
import newton.examples
import numpy as np
import warp as wp

from ..cli import create_parser
from ..config import AUTO_INSERTION, ROBOT_CONTROL, SIMULATION, VIEWER, TaskTargetConfig
from ..controllers.insertion import (
    AttachmentMode,
    AutoCommand,
    AutomaticInsertionController,
    AutoObservation,
    AutoState,
    orientation_error,
    translation_error,
)
from ..controllers.tcp import TCPController
from ..scene.cable import (
    _mounted_sfp_grasp_tcp_target,
)
from ..scene.layout import ManipulationFrames
from ..scene.robot import (
    AIC_GRIPPER_JOINT_NAMES,
    AIC_TOOL_TO_GRIPPER_TCP,
)
from ..scene.visuals import _configure_aic_lighting
from ..utils.labels import find_label_index as _find_label_index
from ..utils.transforms import (
    pose_from_transform,
    transform_from_pose,
)
from ..utils.transforms import (
    transform_from_row as _transform_from_row,
)
from .attachments import (
    VBDAttachmentOwnershipController,
)
from .cameras import CameraPanel, position_camera_window
from .model import ROBOT_ROOT_XFORM, build_simulation_model


@wp.kernel
def _write_automatic_gripper_targets(
    destination_q: wp.array[float],
    destination_indices: wp.array[int],
    target_q: wp.array[float],
):
    index = wp.tid()
    destination_q[destination_indices[index]] = target_q[0]


@wp.kernel
def _set_shape_collisions_enabled(
    shape_flags: wp.array[int],
    shapes: wp.array[int],
    enabled: wp.array[int],
):
    """Toggle shape collision bits without changing the builder collision graph."""
    shape = shapes[wp.tid()]
    if enabled[0] == 0:
        shape_flags[shape] = shape_flags[shape] & ~int(newton.ShapeFlags.COLLIDE_SHAPES)
    else:
        shape_flags[shape] = shape_flags[shape] | int(newton.ShapeFlags.COLLIDE_SHAPES)


def _capture_frame_graph(model: newton.Model, simulate: Callable[[], None], *, enabled: bool):
    """Capture one simulation frame when CUDA graph execution is enabled."""
    if not enabled:
        return None
    with wp.ScopedDevice(model.device):
        with wp.ScopedCapture() as capture:
            simulate()
    if capture.graph is None:
        raise RuntimeError(f"Graph capture failed on device {model.device}")
    return capture.graph


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.auto_enabled = bool(args.auto)
        self.camera_enabled = bool(args.camera)
        self.sim_time = 0.0
        self.fps = SIMULATION.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, int(args.substeps))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.task_target = TaskTargetConfig(
            cable_index=args.cable_index,
            nic_card_index=args.nic_card_index,
            nic_port_index=args.nic_port_index,
        )

        components = build_simulation_model(self.task_target)
        self.model = components.model
        self.solver = components.solver
        self.scene = components.scene
        self.dynamic_cable = components.cable
        self.manipulation_frames = components.manipulation_frames
        self.tool_body = components.tool_body
        self.auto_insert_collision_shapes = components.insertion_collision_shapes
        self.control = self.model.control()
        self.external_joint_control = False
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.tcp_controller = TCPController(
            self.model,
            components.robot_path,
            root_xform=ROBOT_ROOT_XFORM,
        )
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self._setup_attachment_runtime()
        if self.auto_enabled:
            self._setup_automatic_runtime(self.manipulation_frames)
        self._initial_state = self.model.state()
        self._initial_state.assign(self.state_0)
        self._initial_shape_flags = wp.clone(self.model.shape_flags)
        self._reset_key_down = False

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "camera_speed"):
            self.viewer.camera_speed = args.camera_speed
        self.camera_panel = None
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            _configure_aic_lighting(self.viewer)
            self.viewer.set_camera(
                pos=wp.vec3(*VIEWER.camera_position_m),
                pitch=VIEWER.camera_pitch_deg,
                yaw=VIEWER.camera_yaw_deg,
            )
            self.viewer.camera.fov = VIEWER.camera_fov_deg
            if hasattr(self.viewer.camera, "look_at"):
                self.viewer.camera.look_at(wp.vec3(*VIEWER.camera_look_at_m))
            if self.camera_enabled:
                self.camera_panel = CameraPanel(self.model, self.viewer, simulation_rate=self.fps)
                self.viewer.register_ui_callback(position_camera_window, position="free")
            print("TCP controls: I/K X, J/L Y, U/O Z; hold Shift to rotate; R resets the scene")

        use_graph = bool(args.graph_capture) and self.model.device.is_cuda
        self.graph = _capture_frame_graph(self.model, self.simulate, enabled=use_graph)

    def _setup_automatic_runtime(self, frames: ManipulationFrames) -> None:
        """Create the mounted-cable approach toward the reviewed grasp."""
        self.automatic_controller = self._create_automatic_controller(frames)
        self._automatic_failure_logged = False
        self.auto_insert_collision_shape_indices = wp.array(
            self.auto_insert_collision_shapes,
            dtype=int,
            device=self.model.device,
        )
        self.auto_insert_collisions_enabled = wp.array(
            [1],
            dtype=int,
            device=self.model.device,
        )
        starts = self.model.joint_q_start.numpy()
        gripper_coordinates = tuple(
            int(starts[_find_label_index(self.model.joint_label, name)]) for name in AIC_GRIPPER_JOINT_NAMES
        )
        self.control_gripper_coordinate_indices = gripper_coordinates
        self.control_gripper_coordinates = wp.array(
            gripper_coordinates,
            dtype=int,
            device=self.model.device,
        )
        self.auto_gripper_target = wp.array(
            [ROBOT_CONTROL.gripper_open_q],
            dtype=float,
            device=self.model.device,
        )

    def _create_automatic_controller(self, frames: ManipulationFrames) -> AutomaticInsertionController:
        """Create an automatic controller from the current mounted scene."""
        body_q = self.state_0.body_q.numpy()
        mount_sfp_target = _transform_from_row(body_q[self.dynamic_cable.sfp_body])
        grasp_tcp_target = _mounted_sfp_grasp_tcp_target(mount_sfp_target)
        return AutomaticInsertionController(
            frames,
            pose_from_transform(self.tcp_controller.initial_target),
            source_grasp_pose=pose_from_transform(grasp_tcp_target),
            task_target=self.task_target,
        )

    def _setup_attachment_runtime(self) -> None:
        """Initialize the mounted cable ownership shared by all controls."""
        handles = self.dynamic_cable
        self.vbd_attachment_controller = VBDAttachmentOwnershipController(
            solver=self.solver,
            free_root_joint_label=self.model.joint_label[handles.sfp_root_joint],
            mount_joint_label=self.model.joint_label[handles.mount_joint],
            grasp_joint_label=self.model.joint_label[handles.grasp_joint],
            seat_joint_label=self.model.joint_label[handles.seat_joint],
            sc_mount_joint_label=self.model.joint_label[handles.sc_mount_joint],
        )
        self.vbd_attachment_controller.set_mode(AttachmentMode.MOUNTED)

    def _automatic_observation(self) -> AutoObservation:
        """Read finite TCP, SFP-module, and arm state for the controller."""
        handles = self.dynamic_cable
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        joint_q = self.state_0.joint_q.numpy()
        tcp_pose = pose_from_transform(_transform_from_row(body_q[self.tool_body]) * AIC_TOOL_TO_GRIPPER_TCP)
        sfp_body_pose = _transform_from_row(body_q[handles.sfp_body])
        sfp_module_pose = sfp_body_pose * handles.sfp_body_to_module
        arm_q = tuple(float(joint_q[index]) for index in self.tcp_controller.control_arm_coordinate_indices)
        gripper_q = tuple(float(joint_q[index]) for index in self.control_gripper_coordinate_indices)
        return AutoObservation(
            tcp_pose=tcp_pose,
            sfp_pose=pose_from_transform(sfp_module_pose),
            arm_q=arm_q,
            gripper_q=gripper_q,
            finite=bool(np.all(np.isfinite(body_q)) and np.all(np.isfinite(body_qd)) and np.all(np.isfinite(joint_q))),
        )

    def _sync_vbd_attachment_ownership(self, command: AutoCommand) -> None:
        """Route the SFP to exactly one preallocated VBD fixed joint."""
        if (
            command.attachment_mode is AttachmentMode.GRASPED
            and self.vbd_attachment_controller.mode is not AttachmentMode.GRASPED
        ):
            body_q = self.state_0.body_q.numpy()
            self.vbd_attachment_controller.set_mode(
                AttachmentMode.GRASPED,
                tool_pose=_transform_from_row(body_q[self.tool_body]),
                sfp_pose=_transform_from_row(body_q[self.dynamic_cable.sfp_body]),
            )
        else:
            self.vbd_attachment_controller.set_mode(command.attachment_mode)

    def _update_automatic_targets(self) -> None:
        """Update runtime targets for automatic SFP insertion."""
        observation = self._automatic_observation()
        previous_state = self.automatic_controller.state
        if previous_state is AutoState.FAILED:
            previous_target = pose_from_transform(self.tcp_controller.target)
        else:
            previous_target = self.automatic_controller.target_pose
        command = self.automatic_controller.command(observation, dt=self.frame_dt)

        self._sync_vbd_attachment_ownership(command)

        self.tcp_controller.set_target(transform_from_pose(command.tcp_target))
        self.auto_gripper_target.fill_(command.gripper_target)
        if command.state in {
            AutoState.OPEN_GRIPPER,
            AutoState.RETRACT_FROM_PORT,
            AutoState.LIFT_AFTER_RELEASE,
            AutoState.COMPLETE,
        }:
            self.auto_insert_collisions_enabled.fill_(0)

        if command.state is AutoState.FAILED:
            if not self._automatic_failure_logged:
                try:
                    position_error = translation_error(observation.tcp_pose, previous_target)
                    rotation_error = orientation_error(observation.tcp_pose, previous_target)
                except ValueError:
                    position_error = float("nan")
                    rotation_error = float("nan")
                try:
                    seat_target = self.automatic_controller.desired_sfp_target_for(AutoState.INSERT_TO_BOTTOM)
                    sfp_position_error = translation_error(observation.sfp_pose, seat_target)
                    sfp_rotation_error = orientation_error(observation.sfp_pose, seat_target)
                except (AttributeError, KeyError, ValueError):
                    sfp_position_error = float("nan")
                    sfp_rotation_error = float("nan")
                try:
                    body_q = self.state_0.body_q.numpy()
                    tool_pose = _transform_from_row(body_q[self.tool_body])
                    sfp_pose = _transform_from_row(body_q[self.dynamic_cable.sfp_body])
                    grasp_joint = self.vbd_attachment_controller.grasp_joint
                    parent_frame = _transform_from_row(self.vbd_attachment_controller.joint_X_p.numpy()[grasp_joint])
                    child_frame = _transform_from_row(self.vbd_attachment_controller.joint_X_c.numpy()[grasp_joint])
                    actual_grasp = wp.transform_inverse(tool_pose) * sfp_pose
                    expected_grasp = parent_frame * wp.transform_inverse(child_frame)
                    grasp_position_error = translation_error(
                        pose_from_transform(actual_grasp),
                        pose_from_transform(expected_grasp),
                    )
                    grasp_rotation_error = orientation_error(
                        pose_from_transform(actual_grasp),
                        pose_from_transform(expected_grasp),
                    )
                except (AttributeError, IndexError, ValueError):
                    grasp_position_error = float("nan")
                    grasp_rotation_error = float("nan")
                print(
                    f"Automatic insertion failed in {previous_state.name}: "
                    f"translation error {position_error:.6f} m, "
                    f"orientation error {rotation_error:.6f} rad; "
                    f"gripper positions [{', '.join(f'{position:.6f}' for position in observation.gripper_q)}] m; "
                    f"SFP translation error {sfp_position_error:.6f} m, "
                    f"SFP orientation error {sfp_rotation_error:.6f} rad; "
                    f"grasp translation error {grasp_position_error:.6f} m, "
                    f"grasp orientation error {grasp_rotation_error:.6f} rad; "
                    f"{command.failure_message}"
                )
                self._automatic_failure_logged = True
        elif command.state is not previous_state:
            print(f"Automatic insertion: {previous_state.name} -> {command.state.name}")

    def simulate(self):
        if not self.external_joint_control:
            self.tcp_controller.step(self.control, dt=self.frame_dt)
        if self.auto_enabled:
            wp.launch(
                _write_automatic_gripper_targets,
                dim=len(AIC_GRIPPER_JOINT_NAMES),
                inputs=[
                    self.control.joint_target_q,
                    self.control_gripper_coordinates,
                    self.auto_gripper_target,
                ],
                device=self.model.device,
            )
            wp.launch(
                _set_shape_collisions_enabled,
                dim=len(self.auto_insert_collision_shapes),
                inputs=[
                    self.model.shape_flags,
                    self.auto_insert_collision_shape_indices,
                    self.auto_insert_collisions_enabled,
                ],
                device=self.model.device,
            )
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def reset(self) -> None:
        """Restore the complete scene and controller state in place."""
        self.state_0.assign(self._initial_state)
        self.solver.reset(self.state_0, flags=0)
        self.state_1.assign(self.state_0)
        self.control.clear(self.model)
        wp.copy(self.model.shape_flags, self._initial_shape_flags)
        self.contacts.clear()

        self.vbd_attachment_controller.set_mode(AttachmentMode.MOUNTED)
        self.tcp_controller.reset()
        if self.auto_enabled:
            self.automatic_controller = self._create_automatic_controller(self.manipulation_frames)
            self._automatic_failure_logged = False
            self.auto_gripper_target.fill_(ROBOT_CONTROL.gripper_open_q)
            self.auto_insert_collisions_enabled.fill_(1)
        if self.camera_panel is not None:
            self.camera_panel.reset()
        self.sim_time = 0.0
        print("Scene reset")

    def step(self):
        reset_down = bool(hasattr(self.viewer, "is_key_down") and self.viewer.is_key_down("r"))
        if reset_down and not self._reset_key_down:
            self.reset()
        self._reset_key_down = reset_down

        if self.auto_enabled:
            self._update_automatic_targets()
        elif not self.auto_enabled:
            self.tcp_controller.update_manual(
                self.viewer,
                self.state_0,
                tool_body=self.tool_body,
                dt=self.frame_dt,
            )
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        if self.camera_panel is not None:
            self.camera_panel.render(self.state_0)
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify the imported AIC scene remains finite and complete."""
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        assert np.all(np.isfinite(body_q)), "AIC body transforms contain NaN or inf values"
        assert np.all(np.isfinite(body_qd)), "AIC body velocities contain NaN or inf values"
        assert np.all(np.isfinite(joint_q)), "AIC joint positions contain NaN or inf values"
        assert np.all(np.isfinite(joint_qd)), "AIC joint velocities contain NaN or inf values"
        assert len(self.scene.cable_assemblies) == 5
        assert len(self.scene.nic_card_shapes) == 5
        assert len(self.scene.sc_port_shapes) == 5
        if not self.auto_enabled:
            return

        assert self.automatic_controller.state is AutoState.COMPLETE, (
            f"Automatic insertion ended in {self.automatic_controller.state.name}, not COMPLETE"
        )
        sfp_body_pose = _transform_from_row(body_q[self.dynamic_cable.sfp_body])
        sfp_pose = pose_from_transform(sfp_body_pose * self.dynamic_cable.sfp_body_to_module)
        port_bottom = self.manipulation_frames.port_bottom
        sfp_position_error = translation_error(sfp_pose, port_bottom)
        sfp_orientation_error = orientation_error(sfp_pose, port_bottom)
        assert sfp_position_error < AUTO_INSERTION.seat_translation_tolerance_m, (
            f"SFP position error {sfp_position_error:.6f} m exceeds {AUTO_INSERTION.seat_translation_tolerance_m:.6f} m"
        )
        assert sfp_orientation_error < AUTO_INSERTION.seat_orientation_tolerance_rad, (
            f"SFP orientation error {np.rad2deg(sfp_orientation_error):.6f} deg exceeds "
            f"{np.rad2deg(AUTO_INSERTION.seat_orientation_tolerance_rad):.6f} deg"
        )

        gripper_errors = [
            abs(float(joint_q[index]) - ROBOT_CONTROL.gripper_open_q)
            for index in self.control_gripper_coordinate_indices
        ]
        assert max(gripper_errors) < AUTO_INSERTION.gripper_position_tolerance_m, (
            f"Gripper joint error {max(gripper_errors):.6f} m exceeds "
            f"{AUTO_INSERTION.gripper_position_tolerance_m:.6f} m"
        )
        expected_segment_count = len(self.scene.cable_assemblies[self.task_target.cable_index].centerline) - 1
        assert len(self.dynamic_cable.cable_bodies) == expected_segment_count, (
            f"Dynamic cable has {len(self.dynamic_cable.cable_bodies)} segments, expected {expected_segment_count}"
        )


def main() -> None:
    """Run the standalone AIC cable example."""
    parser = create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
