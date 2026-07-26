"""Configure cable geometry, material, contacts, and constraints."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class CableConfig:
    """Configure the flexible cable and connector attachment behavior."""

    segment_length_m: float = 0.01
    radius_m: float = 0.002
    density_kg_m3: float = 4000.0
    color_rgb: tuple[float, float, float] = (1.0, 0.22, 0.0)
    contact_stiffness_n_m: float = 8.0e5
    contact_damping_n_s_m: float = 20.0
    friction_coefficient: float = 0.8
    contact_margin_m: float = 0.0005
    contact_gap_m: float = 0.0005
    stretch_stiffness_n_m: float = 2.0e6
    stretch_damping_n_s_m: float = 5.0
    bend_stiffness: float = 1.0e2
    bend_damping: float = 5.0
    attachment_filter_length_m: float = 0.144
    attachment_stiffness: float = 1.0e7
    endpoint_straight_length_m: float = 0.01
    endpoint_blend_length_m: float = 0.08
    connector_density_kg_m3: float = 1000.0
    connector_contact_margin_m: float = 0.0
    connector_contact_gap_m: float = 0.0

    def __post_init__(self) -> None:
        """Validate physical cable and contact parameters."""
        positive = (
            self.segment_length_m,
            self.radius_m,
            self.density_kg_m3,
            self.contact_stiffness_n_m,
            self.stretch_stiffness_n_m,
            self.attachment_filter_length_m,
            self.attachment_stiffness,
            self.endpoint_straight_length_m,
            self.endpoint_blend_length_m,
            self.connector_density_kg_m3,
        )
        nonnegative = (
            self.contact_damping_n_s_m,
            self.friction_coefficient,
            self.contact_margin_m,
            self.contact_gap_m,
            self.stretch_damping_n_s_m,
            self.bend_stiffness,
            self.bend_damping,
            self.connector_contact_margin_m,
            self.connector_contact_gap_m,
        )
        if not all(isfinite(value) for value in (*positive, *nonnegative, *self.color_rgb)):
            raise ValueError("Cable parameters must be finite")
        if any(value <= 0.0 for value in positive):
            raise ValueError("Cable dimensions, density, and stiffness must be positive")
        if any(value < 0.0 for value in nonnegative):
            raise ValueError("Cable damping, friction, and contact offsets must be nonnegative")
        if len(self.color_rgb) != 3 or any(not 0.0 <= value <= 1.0 for value in self.color_rgb):
            raise ValueError("color_rgb must contain three values between zero and one")


CABLE = CableConfig()
