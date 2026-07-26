# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from aic_newton.config import CABLE, SCENE
from aic_newton.scene.assets import scene_asset_dir
from aic_newton.scene.cable import load_cable_reference
from aic_newton.scene.layout import (
    build_static_cable_assemblies,
    default_layout,
)
from aic_newton.utils.transforms import quat_rotate


class TestScene(unittest.TestCase):
    """Cover the reviewed scene geometry and cable assets."""

    def test_build_configured_cables_with_aligned_endpoints(self):
        """Build configured cable pairs with bounded segments and aligned endpoints."""
        assemblies = build_static_cable_assemblies(
            default_layout(),
            load_cable_reference(scene_asset_dir() / "cable_reference.json"),
        )

        self.assertEqual(len(assemblies), len(SCENE.cable_pair_translations_m))
        for assembly in assemblies:
            points = np.asarray(assembly.centerline)
            segments = np.diff(points, axis=0)
            lengths = np.linalg.norm(segments, axis=1)
            self.assertLessEqual(float(np.max(lengths[1:-1])), CABLE.segment_length_m * 1.01)
            np.testing.assert_allclose(
                segments[0],
                CABLE.endpoint_straight_length_m
                * np.asarray(quat_rotate(assembly.lc_plug_quat_xyzw, (0.0, -1.0, 0.0))),
                atol=1.0e-6,
            )
            np.testing.assert_allclose(
                -segments[-1],
                CABLE.endpoint_straight_length_m
                * np.asarray(quat_rotate(assembly.sc_plug_quat_xyzw, (-1.0, 0.0, 0.0))),
                atol=1.0e-6,
            )


if __name__ == "__main__":
    unittest.main()
