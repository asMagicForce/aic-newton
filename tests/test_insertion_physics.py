"""Physical regression coverage for contact-safe automatic insertion."""

import math
import subprocess
import sys
from pathlib import Path

import newton
import warp as wp

from aic_newton.cli import create_parser
from aic_newton.config import AUTO_INSERTION
from aic_newton.controllers.insertion import AutoState
from aic_newton.controllers.trajectory import orientation_error, translation_error
from aic_newton.scene.robot import AIC_TOOL_TO_GRIPPER_TCP
from aic_newton.simulation.application import Example
from aic_newton.simulation.attachments import AttachmentMode
from aic_newton.utils.transforms import pose_from_transform, transform_from_row


def _assert_automatic_insertion_preserves_the_grasp_and_reaches_the_port() -> None:
    """Complete insertion without disabling contact, slipping the grasp, or snapping the SFP."""
    args = create_parser().parse_args(["--viewer", "null", "--auto"])
    example = Example(newton.viewer.ViewerNull(num_frames=1), args)

    def tcp_to_sfp():
        body_q = example.state_0.body_q.numpy()
        tcp = transform_from_row(body_q[example.tool_body]) * AIC_TOOL_TO_GRIPPER_TCP
        handles = example.dynamic_cable
        sfp = transform_from_row(body_q[handles.sfp_body]) * handles.sfp_body_to_module
        return pose_from_transform(wp.transform_inverse(tcp) * sfp)

    captured = None
    maximum_translation = 0.0
    maximum_orientation = 0.0
    maximum_translation_state = None
    maximum_orientation_state = None
    maximum_insert_step = 0.0
    previous_insert_sfp = None
    observed_insert_collisions = False
    grasped_states = {
        AutoState.EXTRACT_FROM_MOUNT,
        AutoState.LIFT_AFTER_EXTRACTION,
        AutoState.TRANSFER_ABOVE_PORT,
        AutoState.ALIGN_WITH_PORT,
        AutoState.INSERT_TO_BOTTOM,
    }
    for _ in range(3600):
        previous_state = example.automatic_controller.state
        example.step()
        if example.automatic_controller.state is AutoState.INSERT_TO_BOTTOM:
            observed_insert_collisions = True
            current_sfp = example._automatic_observation().sfp_pose
            if previous_insert_sfp is not None:
                maximum_insert_step = max(
                    maximum_insert_step,
                    translation_error(current_sfp, previous_insert_sfp),
                )
            previous_insert_sfp = current_sfp
            shape_flags = example.model.shape_flags.numpy()
            assert all(
                int(shape_flags[shape]) & int(newton.ShapeFlags.COLLIDE_SHAPES)
                for shape in example.auto_insert_collision_shapes
            ), "SFP collision shapes must remain enabled throughout insertion"
        if (
            example.vbd_attachment_controller.mode is AttachmentMode.GRASPED
            and previous_state in grasped_states
        ):
            current = tcp_to_sfp()
            if captured is None:
                captured = current
            current_translation = translation_error(current, captured)
            current_orientation = orientation_error(current, captured)
            if current_translation > maximum_translation:
                maximum_translation = current_translation
                maximum_translation_state = previous_state
            if current_orientation > maximum_orientation:
                maximum_orientation = current_orientation
                maximum_orientation_state = previous_state
        if example.automatic_controller.state in {AutoState.COMPLETE, AutoState.FAILED}:
            break

    assert captured is not None
    assert observed_insert_collisions
    assert example.automatic_controller.state is AutoState.COMPLETE
    assert maximum_translation < 0.001, (
        f"maximum grasp translation error {maximum_translation:.6f} m in {maximum_translation_state}"
    )
    assert maximum_orientation < math.radians(0.2), (
        f"maximum grasp orientation error {maximum_orientation:.6f} rad in {maximum_orientation_state}"
    )
    assert maximum_insert_step < 0.002, f"SFP snapped {maximum_insert_step:.6f} m in one frame"
    final = example._automatic_observation()
    port = example.manipulation_frames.port_bottom
    assert translation_error(final.sfp_pose, port) <= AUTO_INSERTION.seat_translation_tolerance_m
    assert orientation_error(final.sfp_pose, port) <= AUTO_INSERTION.seat_orientation_tolerance_rad


def test_automatic_insertion_preserves_the_grasp_and_reaches_the_port() -> None:
    """Run graph capture in a clean process, isolated from camera-stream tests."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        check=False,
        capture_output=True,
        text=True,
        timeout=120.0,
    )

    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _assert_automatic_insertion_preserves_the_grasp_and_reaches_the_port()
