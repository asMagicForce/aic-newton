"""Configure task-board assembly geometry."""

from dataclasses import dataclass
from math import hypot, isfinite


@dataclass(frozen=True)
class SceneConfig:
    """Configure task-board layout and nominal connector poses."""

    board_xyz_m: tuple[float, float, float] = (0.25617, 0.047549, 1.14)
    board_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 2.77239)
    cable_pair_rails: tuple[int, ...] = (0, 0, 0, 1, 1)
    cable_pair_translations_m: tuple[float, ...] = (-0.080948, -0.012195, 0.07793, -0.009757, 0.083624)
    nic_translations_m: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    sc_port_translations_m: tuple[float, ...] = (-0.07, 0.0, 0.07, -0.04, 0.04)
    # Align the SFP with the 45-degree guide while retaining 0.5 mm normal clearance.
    mount_to_sfp_xyz_m: tuple[float, float, float] = (0.02915663, 0.0, 0.02168374)
    mount_to_sfp_quat_xyzw: tuple[float, float, float, float] = (
        0.65328148,
        0.65328148,
        0.27059805,
        0.27059805,
    )

    def __post_init__(self) -> None:
        """Validate task-board layout and the mount-to-SFP transform."""
        if len(self.board_xyz_m) != 3 or len(self.board_rpy_rad) != 3:
            raise ValueError("Task-board pose must contain XYZ and RPY values")
        layout_lengths = {
            len(self.cable_pair_rails),
            len(self.cable_pair_translations_m),
            len(self.nic_translations_m),
            len(self.sc_port_translations_m),
        }
        if layout_lengths != {5}:
            raise ValueError("AIC layout must define five cable pairs, NIC cards, and SC ports")
        if any(rail not in (0, 1) for rail in self.cable_pair_rails):
            raise ValueError("cable_pair_rails entries must be zero or one")
        values = (
            *self.board_xyz_m,
            *self.board_rpy_rad,
            *self.cable_pair_translations_m,
            *self.nic_translations_m,
            *self.sc_port_translations_m,
            *self.mount_to_sfp_xyz_m,
            *self.mount_to_sfp_quat_xyzw,
        )
        if len(self.mount_to_sfp_xyz_m) != 3 or len(self.mount_to_sfp_quat_xyzw) != 4:
            raise ValueError("mount-to-SFP transform must contain XYZ and XYZW values")
        if not all(isfinite(value) for value in values):
            raise ValueError("Scene layout and transforms must contain only finite values")
        if abs(hypot(*self.mount_to_sfp_quat_xyzw) - 1.0) > 1.0e-7:
            raise ValueError("mount_to_sfp_quat_xyzw must be normalized")


SCENE = SceneConfig()
