"""Physical regression coverage for the complete automatic insertion."""

import math
import subprocess
import sys
from pathlib import Path

import newton
import warp as wp

from aic_newton.cli import create_parser
from aic_newton.controllers.insertion import AutoState
from aic_newton.controllers.trajectory import orientation_error, translation_error
from aic_newton.scene.robot import AIC_TOOL_TO_GRIPPER_TCP
from aic_newton.simulation.application import Example
from aic_newton.simulation.attachments import AttachmentMode
from aic_newton.utils.transforms import pose_from_transform, transform_from_row


def _assert_automatic_insertion_preserves_the_grasp_and_physically_reaches_the_port() -> None:
    """Catch URDF migration changing grasp dynamics or masking a failed insert."""
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
    free_space_states = {
        AutoState.EXTRACT_FROM_MOUNT,
        AutoState.LIFT_AFTER_EXTRACTION,
        AutoState.TRANSFER_ABOVE_PORT,
        AutoState.ALIGN_WITH_PORT,
    }
    for _ in range(3600):
        previous_state = example.automatic_controller.state
        example.step()
        if (
            example.vbd_attachment_controller.mode is AttachmentMode.GRASPED
            and previous_state in free_space_states
        ):
            current = tcp_to_sfp()
            if captured is None:
                captured = current
            maximum_translation = max(
                maximum_translation, translation_error(current, captured)
            )
            maximum_orientation = max(
                maximum_orientation, orientation_error(current, captured)
            )
        if example.automatic_controller.state in {AutoState.COMPLETE, AutoState.FAILED}:
            break

    assert captured is not None
    assert example.automatic_controller.state is AutoState.COMPLETE
    assert maximum_translation < 0.001
    assert maximum_orientation < math.radians(0.2)
    final = example._automatic_observation()
    port = example.manipulation_frames.port_bottom
    assert translation_error(final.sfp_pose, port) < 0.001
    assert orientation_error(final.sfp_pose, port) < math.radians(1.0)


def test_automatic_insertion_preserves_the_grasp_and_physically_reaches_the_port() -> None:
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
    _assert_automatic_insertion_preserves_the_grasp_and_physically_reaches_the_port()
