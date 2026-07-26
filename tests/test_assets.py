"""Test repository-owned AIC asset resolution."""

import unittest

from aic_newton.scene.assets import mjcf_dir, repository_root, scene_asset_dir, visual_model_dir


class TestAssets(unittest.TestCase):
    """Verify runtime assets are fully vendored."""

    def test_resolve_runtime_assets_inside_repository(self):
        """Resolve the required simulation assets below this repository."""
        root = repository_root()
        paths = (
            mjcf_dir() / "aic_robot.xml",
            mjcf_dir() / "aic_world.xml",
            visual_model_dir() / "LC Plug" / "lc_plug_visual.glb",
            scene_asset_dir() / "cable_reference.json",
        )

        for path in paths:
            with self.subTest(path=path):
                self.assertTrue(path.is_file())
                self.assertTrue(path.is_relative_to(root))


if __name__ == "__main__":
    unittest.main()
