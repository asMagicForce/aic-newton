# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from aic_newton.config import AUTO_INSERTION, ROBOT_CONTROL
from aic_newton.controllers.insertion import (
    AutomaticInsertionController,
    AutoObservation,
    AutoState,
)
from aic_newton.controllers.trajectory import orientation_error, translation_error
from aic_newton.scene.assets import scene_asset_dir, visual_model_dir
from aic_newton.scene.cable import load_cable_reference
from aic_newton.scene.layout import build_static_cable_assemblies, default_layout, manipulation_frames
from aic_newton.simulation.attachments import AttachmentMode
from aic_newton.utils.transforms import PoseTuple, compose_pose, inverse_pose, quat_rotate


class TestAutomaticInsertionController(unittest.TestCase):
    """Cover the automatic task sequence and its safety gates."""

    def setUp(self):
        """Build the reviewed frames and a deterministic home state."""
        layout = default_layout()
        assembly = build_static_cable_assemblies(
            layout,
            load_cable_reference(scene_asset_dir() / "cable_reference.json"),
        )[0]
        self.frames = manipulation_frames(layout, assembly, visual_model_dir() / "NIC Card" / "model.sdf")
        self.home_pose = PoseTuple((0.15, 0.15, 1.35), (0.0, 0.0, 0.0, 1.0))
        self.home_arm_q = ROBOT_CONTROL.home_q
        self.tracked_tcp_targets: dict[int, PoseTuple] = {}
        self.tracked_sfp_poses: dict[int, PoseTuple] = {}

    def _controller(self) -> AutomaticInsertionController:
        """Create the controller under test."""
        controller = AutomaticInsertionController(self.frames, self.home_pose)
        self.tracked_tcp_targets[id(controller)] = self.home_pose
        self.tracked_sfp_poses[id(controller)] = self.frames.sfp_module
        return controller

    def _gripper_target(self, state: AutoState) -> float:
        """Return the expected gripper target for a state."""
        if state in {
            AutoState.CLOSE_GRIPPER,
            AutoState.EXTRACT_FROM_MOUNT,
            AutoState.LIFT_AFTER_EXTRACTION,
            AutoState.TRANSFER_ABOVE_PORT,
            AutoState.ALIGN_WITH_PORT,
            AutoState.INSERT_TO_BOTTOM,
        }:
            return ROBOT_CONTROL.gripper_closed_q
        return ROBOT_CONTROL.gripper_open_q

    def _observation(
        self,
        controller: AutomaticInsertionController,
        *,
        tcp_pose: PoseTuple | None = None,
        sfp_pose: PoseTuple | None = None,
        finite: bool = True,
        gripper_q: tuple[float, float] | None = None,
    ) -> AutoObservation:
        """Build an observation at the current state's target."""
        target = self._gripper_target(controller.state)
        return AutoObservation(
            tcp_pose=controller.target_pose if tcp_pose is None else tcp_pose,
            sfp_pose=self.frames.sfp_module if sfp_pose is None else sfp_pose,
            arm_q=self.home_arm_q,
            gripper_q=(target, target) if gripper_q is None else gripper_q,
            finite=finite,
        )

    def _advance(
        self,
        controller: AutomaticInsertionController,
        *,
        sfp_pose: PoseTuple | None = None,
        gripper_q: tuple[float, float] | None = None,
    ):
        """Advance one state after tracking its final target."""
        start = controller.state
        if sfp_pose is None:
            try:
                sfp_pose = controller.desired_sfp_target_for(controller.state)
            except ValueError:
                sfp_pose = self.frames.sfp_module
        command = controller.command(
            self._observation(
                controller,
                tcp_pose=self.tracked_tcp_targets[id(controller)],
                sfp_pose=self.tracked_sfp_poses[id(controller)],
                gripper_q=gripper_q,
            ),
            dt=controller.state_duration,
        )
        if controller.state is start:
            command = controller.command(
                self._observation(controller, sfp_pose=sfp_pose, gripper_q=gripper_q),
                dt=1.0e-6,
            )
        self.tracked_tcp_targets[id(controller)] = command.tcp_target
        self.tracked_sfp_poses[id(controller)] = sfp_pose
        return command

    def _advance_to(self, controller: AutomaticInsertionController, target: AutoState) -> None:
        """Advance through public commands until the requested state."""
        for _ in AutoState:
            if controller.state is target:
                return
            self._advance(controller)
        self.fail(f"Controller did not reach {target.name}")

    def test_complete_nominal_sequence_and_attachment_handoffs(self):
        """Complete extraction and insertion with mount, grasp, and seat ownership."""
        controller = self._controller()
        ownership = [AttachmentMode.MOUNTED]

        for _ in AutoState:
            if controller.state in (AutoState.COMPLETE, AutoState.FAILED):
                break
            sfp_pose = (
                self.frames.port_bottom
                if controller.state in (AutoState.INSERT_TO_BOTTOM, AutoState.OPEN_GRIPPER)
                else None
            )
            command = self._advance(controller, sfp_pose=sfp_pose)
            ownership.append(command.attachment_mode)
        else:
            self.fail("Controller did not reach a terminal state")

        self.assertEqual(controller.state, AutoState.COMPLETE)
        self.assertIn(AttachmentMode.GRASPED, ownership)
        self.assertEqual(ownership[-1], AttachmentMode.SEATED)

    def test_derive_extraction_and_port_targets_after_grasp(self):
        """Move the SFP along its long axis before targeting the NIC port."""
        controller = self._controller()
        self._advance_to(controller, AutoState.EXTRACT_FROM_MOUNT)

        extraction = controller.desired_sfp_target_for(AutoState.EXTRACT_FROM_MOUNT)
        axis = np.asarray(quat_rotate(self.frames.sfp_module.quat_xyzw, (0.0, 1.0, 0.0)))
        np.testing.assert_allclose(
            np.asarray(extraction.xyz) - np.asarray(self.frames.sfp_module.xyz),
            AUTO_INSERTION.sfp_extraction_distance_m * axis,
            atol=1.0e-6,
        )
        port_axis = np.asarray(self.frames.port_entrance.xyz) - np.asarray(self.frames.port_bottom.xyz)
        port_axis /= np.linalg.norm(port_axis)
        align = controller.desired_sfp_target_for(AutoState.ALIGN_WITH_PORT)
        np.testing.assert_allclose(
            np.asarray(align.xyz) - np.asarray(self.frames.port_entrance.xyz),
            AUTO_INSERTION.align_port_axis_clearance_m * port_axis,
            atol=1.0e-6,
        )
        self.assertEqual(align.quat_xyzw, self.frames.port_entrance.quat_xyzw)
        self.assertEqual(controller.desired_sfp_target_for(AutoState.INSERT_TO_BOTTOM), self.frames.port_bottom)

    def test_refines_grasp_after_the_physics_attachment_becomes_active(self):
        """Use the post-attachment relation when ownership changes on a later physics tick."""
        controller = self._controller()
        self._advance_to(controller, AutoState.EXTRACT_FROM_MOUNT)
        tcp_pose = self.tracked_tcp_targets[id(controller)]
        previous_sfp = self.tracked_sfp_poses[id(controller)]
        attached_sfp = PoseTuple(
            (previous_sfp.xyz[0] + 0.0015, previous_sfp.xyz[1], previous_sfp.xyz[2]),
            previous_sfp.quat_xyzw,
        )
        observed_tool_to_sfp = compose_pose(inverse_pose(tcp_pose), attached_sfp)

        controller.command(
            self._observation(
                controller,
                tcp_pose=tcp_pose,
                sfp_pose=attached_sfp,
            ),
            dt=1.0e-6,
        )

        self.assertIsNotNone(controller.tool_to_sfp)
        self.assertLess(
            translation_error(controller.tool_to_sfp, observed_tool_to_sfp),
            1.0e-9,
        )
        transfer_tcp = controller.target_for(AutoState.TRANSFER_ABOVE_PORT)
        projected_sfp = compose_pose(transfer_tcp, controller.tool_to_sfp)
        desired_sfp = controller.desired_sfp_target_for(AutoState.TRANSFER_ABOVE_PORT)
        self.assertLess(translation_error(projected_sfp, desired_sfp), 1.0e-9)

    def test_require_both_fingers_before_grasp_handoff(self):
        """Keep mount ownership until both fingers reach the closed target."""
        controller = self._controller()
        self._advance_to(controller, AutoState.CLOSE_GRIPPER)

        command = controller.command(
            self._observation(
                controller,
                gripper_q=(
                    ROBOT_CONTROL.gripper_closed_q + 1.1 * AUTO_INSERTION.gripper_position_tolerance_m,
                    ROBOT_CONTROL.gripper_closed_q,
                ),
            ),
            dt=controller.state_duration,
        )

        self.assertEqual(command.state, AutoState.CLOSE_GRIPPER)
        self.assertEqual(command.attachment_mode, AttachmentMode.MOUNTED)

    def test_transfer_waits_for_observed_sfp_alignment(self):
        """Correct the TCP from observed grasp geometry when only it reached target."""
        controller = self._controller()
        self._advance_to(controller, AutoState.TRANSFER_ABOVE_PORT)
        desired = controller.desired_sfp_target_for(AutoState.TRANSFER_ABOVE_PORT)
        transfer_tcp = controller.target_pose
        displaced = PoseTuple(
            (desired.xyz[0] + 0.002, desired.xyz[1], desired.xyz[2]),
            desired.quat_xyzw,
        )
        observed_tool_to_sfp = compose_pose(inverse_pose(transfer_tcp), displaced)

        command = self._advance(controller, sfp_pose=displaced)

        self.assertEqual(command.state, AutoState.TRANSFER_ABOVE_PORT)
        self.assertEqual(command.attachment_mode, AttachmentMode.GRASPED)
        corrected_sfp = compose_pose(controller.target_pose, observed_tool_to_sfp)
        self.assertLess(translation_error(corrected_sfp, desired), 1.0e-9)
        self.assertLess(orientation_error(corrected_sfp, desired), 1.0e-9)

    def test_entering_align_rebases_port_targets_from_the_observed_grasp(self):
        """Remove accumulated grasp bias before commanding the narrow port approach."""
        controller = self._controller()
        self._advance_to(controller, AutoState.TRANSFER_ABOVE_PORT)
        transfer_tcp = controller.target_pose
        desired_transfer = controller.desired_sfp_target_for(AutoState.TRANSFER_ABOVE_PORT)
        observed_sfp = PoseTuple(
            (desired_transfer.xyz[0] + 0.0005, desired_transfer.xyz[1], desired_transfer.xyz[2]),
            desired_transfer.quat_xyzw,
        )
        observed_tool_to_sfp = compose_pose(inverse_pose(transfer_tcp), observed_sfp)

        command = self._advance(controller, sfp_pose=observed_sfp)

        self.assertEqual(command.state, AutoState.ALIGN_WITH_PORT)
        projected_sfp = compose_pose(controller.target_pose, observed_tool_to_sfp)
        desired_align = controller.desired_sfp_target_for(AutoState.ALIGN_WITH_PORT)
        self.assertLess(translation_error(projected_sfp, desired_align), 1.0e-9)
        self.assertLess(orientation_error(projected_sfp, desired_align), 1.0e-9)

    def test_align_waits_for_observed_sfp_alignment(self):
        """Do not start insertion while the actual module remains laterally offset."""
        controller = self._controller()
        self._advance_to(controller, AutoState.ALIGN_WITH_PORT)
        desired = controller.desired_sfp_target_for(AutoState.ALIGN_WITH_PORT)
        displaced = PoseTuple(
            (desired.xyz[0] + 0.002, desired.xyz[1], desired.xyz[2]),
            desired.quat_xyzw,
        )

        command = self._advance(controller, sfp_pose=displaced)

        self.assertEqual(command.state, AutoState.ALIGN_WITH_PORT)
        self.assertEqual(command.attachment_mode, AttachmentMode.GRASPED)

    def test_align_refines_the_observed_grasp_before_insertion(self):
        """Track one corrected TCP target before accepting an in-tolerance port alignment."""
        controller = self._controller()
        self._advance_to(controller, AutoState.ALIGN_WITH_PORT)
        desired = controller.desired_sfp_target_for(AutoState.ALIGN_WITH_PORT)
        nominal_align_tcp = controller.target_pose
        observed_sfp = PoseTuple(
            (desired.xyz[0] + 0.0005, desired.xyz[1], desired.xyz[2]),
            desired.quat_xyzw,
        )

        command = self._advance(controller, sfp_pose=observed_sfp)

        self.assertEqual(command.state, AutoState.ALIGN_WITH_PORT)
        observed_tool_to_sfp = compose_pose(inverse_pose(nominal_align_tcp), observed_sfp)
        projected_sfp = compose_pose(command.tcp_target, observed_tool_to_sfp)
        self.assertLess(translation_error(projected_sfp, desired), 1.0e-9)
        self.assertLess(orientation_error(projected_sfp, desired), 1.0e-9)

    def test_insertion_guard_uses_the_grasp_relation_at_port_entry(self):
        """Measure insertion slip from the settled alignment, not free-space transport."""
        controller = self._controller()
        self._advance_to(controller, AutoState.ALIGN_WITH_PORT)
        desired = controller.desired_sfp_target_for(AutoState.ALIGN_WITH_PORT)
        observed_sfp = PoseTuple(
            (desired.xyz[0] + 0.0005, desired.xyz[1], desired.xyz[2]),
            desired.quat_xyzw,
        )
        first = controller.command(
            self._observation(
                controller,
                tcp_pose=controller.target_pose,
                sfp_pose=observed_sfp,
            ),
            dt=controller.state_duration,
        )
        expected_tool_to_sfp = compose_pose(inverse_pose(first.tcp_target), desired)

        second = controller.command(
            self._observation(
                controller,
                tcp_pose=first.tcp_target,
                sfp_pose=desired,
            ),
            dt=1.0e-6,
        )

        self.assertEqual(second.state, AutoState.INSERT_TO_BOTTOM)
        self.assertIsNotNone(controller.tool_to_sfp)
        self.assertLess(
            translation_error(controller.tool_to_sfp, expected_tool_to_sfp),
            1.0e-9,
        )
        self.assertLess(
            orientation_error(controller.tool_to_sfp, expected_tool_to_sfp),
            1.0e-9,
        )

    def test_refined_align_rejects_a_0_2_mm_lateral_error(self):
        """Keep refining when the SFP center remains outside the modeled port clearance."""
        controller = self._controller()
        self._advance_to(controller, AutoState.ALIGN_WITH_PORT)
        desired = controller.desired_sfp_target_for(AutoState.ALIGN_WITH_PORT)
        first = self._advance(controller, sfp_pose=desired)
        self.assertEqual(first.state, AutoState.ALIGN_WITH_PORT)
        port_axis = np.asarray(self.frames.port_entrance.xyz) - np.asarray(self.frames.port_bottom.xyz)
        port_axis /= np.linalg.norm(port_axis)
        reference_axis = np.asarray((0.0, 0.0, 1.0))
        if abs(float(port_axis @ reference_axis)) > 0.9:
            reference_axis = np.asarray((1.0, 0.0, 0.0))
        lateral_axis = np.cross(port_axis, reference_axis)
        lateral_axis /= np.linalg.norm(lateral_axis)
        displaced = PoseTuple(
            tuple(np.asarray(desired.xyz) + 0.0002 * lateral_axis),
            desired.quat_xyzw,
        )

        command = controller.command(
            self._observation(
                controller,
                tcp_pose=first.tcp_target,
                sfp_pose=displaced,
            ),
            dt=1.0e-6,
        )

        self.assertEqual(command.state, AutoState.ALIGN_WITH_PORT)
        self.assertEqual(command.attachment_mode, AttachmentMode.GRASPED)

    def test_insert_waits_for_actual_port_bottom_before_seating(self):
        """Do not use SEATED ownership to hide an incomplete insertion."""
        controller = self._controller()
        self._advance_to(controller, AutoState.INSERT_TO_BOTTOM)
        desired = controller.desired_sfp_target_for(AutoState.INSERT_TO_BOTTOM)
        displaced = PoseTuple(
            (desired.xyz[0] + 0.004, desired.xyz[1], desired.xyz[2]),
            desired.quat_xyzw,
        )

        command = self._advance(controller, sfp_pose=displaced)

        self.assertEqual(command.state, AutoState.INSERT_TO_BOTTOM)
        self.assertEqual(command.attachment_mode, AttachmentMode.GRASPED)

    def test_insert_holds_the_tcp_target_when_the_grasped_sfp_is_blocked(self):
        """Do not keep advancing the robot after contact separates the tool and SFP."""
        controller = self._controller()
        self._advance_to(controller, AutoState.INSERT_TO_BOTTOM)
        blocked_sfp = controller.desired_sfp_target_for(AutoState.ALIGN_WITH_PORT)

        first = controller.command(
            self._observation(
                controller,
                tcp_pose=self.tracked_tcp_targets[id(controller)],
                sfp_pose=blocked_sfp,
            ),
            dt=0.25 * controller.state_duration,
        )
        second = controller.command(
            self._observation(
                controller,
                tcp_pose=first.tcp_target,
                sfp_pose=blocked_sfp,
            ),
            dt=0.1,
        )

        self.assertEqual(second.state, AutoState.INSERT_TO_BOTTOM)
        projected_sfp = compose_pose(second.tcp_target, controller.tool_to_sfp)
        self.assertLess(translation_error(projected_sfp, blocked_sfp), 1.0e-9)
        self.assertLess(orientation_error(projected_sfp, blocked_sfp), 1.0e-9)
        self.assertEqual(second.attachment_mode, AttachmentMode.GRASPED)

    def test_seated_contact_is_not_classified_as_a_blocked_grasp(self):
        """Accept the modeled port end stop before applying the blocked-insertion timeout."""
        controller = self._controller()
        self._advance_to(controller, AutoState.INSERT_TO_BOTTOM)
        seated_sfp = controller.desired_sfp_target_for(AutoState.INSERT_TO_BOTTOM)
        target_tcp = controller.target_pose
        offset_tcp = PoseTuple(
            (target_tcp.xyz[0] + 0.001, target_tcp.xyz[1], target_tcp.xyz[2]),
            target_tcp.quat_xyzw,
        )
        observation = self._observation(
            controller,
            tcp_pose=offset_tcp,
            sfp_pose=seated_sfp,
        )

        controller.command(observation, dt=0.1)
        command = controller.command(observation, dt=AUTO_INSERTION.trajectory_stall_timeout_s)

        self.assertNotEqual(command.state, AutoState.FAILED)

    def test_fail_safely_on_timeout_or_non_finite_state(self):
        """Enter the failed hold state on timeout or non-finite observations."""
        controller = self._controller()
        offset = PoseTuple((self.home_pose.xyz[0] + 0.01, *self.home_pose.xyz[1:]), self.home_pose.quat_xyzw)
        observation = AutoObservation(
            offset,
            self.frames.sfp_module,
            self.home_arm_q,
            (ROBOT_CONTROL.gripper_open_q, ROBOT_CONTROL.gripper_open_q),
        )
        command = controller.command(
            observation,
            dt=controller.state_duration + AUTO_INSERTION.state_timeout_margin + 1.0,
        )
        self.assertEqual(command.state, AutoState.FAILED)
        self.assertEqual(command.attachment_mode, AttachmentMode.FAILED)

        controller = self._controller()
        command = controller.command(self._observation(controller, finite=False), dt=0.01)
        self.assertEqual(command.state, AutoState.FAILED)
        self.assertIn("non-finite", command.failure_message)


if __name__ == "__main__":
    unittest.main()
