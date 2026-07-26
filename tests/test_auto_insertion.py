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
from aic_newton.scene.assets import scene_asset_dir, visual_model_dir
from aic_newton.scene.cable import load_cable_reference
from aic_newton.scene.layout import build_static_cable_assemblies, default_layout, manipulation_frames
from aic_newton.simulation.attachments import AttachmentMode
from aic_newton.utils.transforms import PoseTuple, quat_rotate


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

    def _controller(self) -> AutomaticInsertionController:
        """Create the controller under test."""
        controller = AutomaticInsertionController(self.frames, self.home_pose)
        self.tracked_tcp_targets[id(controller)] = self.home_pose
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
        command = controller.command(
            self._observation(
                controller,
                tcp_pose=self.tracked_tcp_targets[id(controller)],
                sfp_pose=sfp_pose,
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
        return command

    def _advance_to(self, controller: AutomaticInsertionController, target: AutoState) -> None:
        """Advance through public commands until the requested state."""
        for _ in AutoState:
            if controller.state is target:
                return
            self._advance(controller, sfp_pose=self.frames.sfp_module)
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
