# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import newton
import numpy as np
import warp as wp

from aic_newton.cli import create_parser
from aic_newton.config import MANUAL_TCP
from aic_newton.controllers.insertion import AutoState
from aic_newton.controllers.manual_tcp import integrate_tcp_target, read_tcp_keyboard
from aic_newton.simulation.application import Example


class KeyViewer:
    """Provide viewer key state for manual-control tests."""

    def __init__(self, *keys: str):
        self.keys = set(keys)

    def is_key_down(self, key: str | int) -> bool:
        """Report whether a key is active."""
        return key in self.keys


class TestAicNewton(unittest.TestCase):
    """Cover the central simulation assembly and control paths."""

    def test_example_uses_runtime_task_indices(self):
        """Build the selected cable and NIC target from runtime indices."""
        args = create_parser().parse_args(
            [
                "--viewer",
                "null",
                "--auto",
                "--no-graph-capture",
                "--cable-index",
                "1",
                "--nic-card-index",
                "1",
                "--nic-port-index",
                "1",
            ]
        )

        example = Example(newton.viewer.ViewerNull(num_frames=1), args)

        self.assertEqual(example.manipulation_frames.cable_name, "cable_1")
        self.assertEqual(example.task_target.cable_index, 1)
        self.assertEqual(example.task_target.nic_card_index, 1)
        self.assertEqual(example.task_target.nic_port_index, 1)

    def test_reset_restores_complete_automatic_scene(self):
        """Restore physics, ownership, controllers, and time in automatic mode."""
        args = create_parser().parse_args(["--viewer", "null", "--auto", "--no-graph-capture"])
        example = Example(newton.viewer.ViewerNull(num_frames=1), args)
        initial_body_q = example.state_0.body_q.numpy()
        initial_joint_q = example.state_0.joint_q.numpy()

        example.state_0.body_qd.assign(np.ones_like(example.state_0.body_qd.numpy()))
        example.state_0.joint_q.assign(initial_joint_q + 0.1)
        example.automatic_controller.state = AutoState.COMPLETE
        example.sim_time = 3.0

        example.reset()

        np.testing.assert_allclose(example.state_0.body_q.numpy(), initial_body_q)
        np.testing.assert_allclose(example.state_0.joint_q.numpy(), initial_joint_q)
        np.testing.assert_allclose(example.state_0.body_qd.numpy(), 0.0)
        self.assertEqual(example.automatic_controller.state, AutoState.HOME)
        self.assertEqual(example.sim_time, 0.0)

    def test_manual_tcp_keyboard_translation_and_rotation(self):
        """Map and integrate manual TCP translation and rotation commands."""
        linear, angular, reset = read_tcp_keyboard(KeyViewer("i"))
        np.testing.assert_allclose(linear, (1.0, 0.0, 0.0))
        np.testing.assert_allclose(angular, (0.0, 0.0, 0.0))
        self.assertFalse(reset)

        linear, angular, _ = read_tcp_keyboard(KeyViewer("shift", "u"))
        np.testing.assert_allclose(linear, (0.0, 0.0, 0.0))
        np.testing.assert_allclose(angular, (0.0, 0.0, 1.0))

        target = integrate_tcp_target(
            wp.transform_identity(),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 0.0, 0.0),
            1.0,
        )
        np.testing.assert_allclose(target.p, (MANUAL_TCP.translation_speed_m_s, 0.0, 0.0), atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()
