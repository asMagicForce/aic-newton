"""Configure Viewer presentation and task-scene lighting."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ViewerConfig:
    """Configure the initial view, navigation, and lighting."""

    camera_position_m: tuple[float, float, float] = (0.64, -0.64, 2.15)
    camera_pitch_deg: float = -18.0
    camera_yaw_deg: float = 138.0
    camera_fov_deg: float = 65.0
    camera_look_at_m: tuple[float, float, float] = (-0.05, 0.0, 1.45)
    camera_speed_m_s: float = 0.5
    shadow_radius_m: float = 6.0
    shadow_extents_m: float = 4.0
    diffuse_scale: float = 1.0
    specular_scale: float = 0.55
    exposure: float = 1.45
    light_color_rgb: tuple[float, float, float] = (1.55, 1.55, 1.55)
    ambient_sky_rgb: tuple[float, float, float] = (0.78, 0.78, 0.80)
    ambient_ground_rgb: tuple[float, float, float] = (0.58, 0.58, 0.60)
    sky_upper_rgb: tuple[float, float, float] = (0.15, 0.15, 0.15)
    sky_lower_rgb: tuple[float, float, float] = (0.15, 0.15, 0.15)
    sun_direction: tuple[float, float, float] = (0.08, -0.05, 1.0)

    def __post_init__(self) -> None:
        """Validate view and lighting parameters."""
        vectors = (
            self.camera_position_m,
            self.camera_look_at_m,
            self.light_color_rgb,
            self.ambient_sky_rgb,
            self.ambient_ground_rgb,
            self.sky_upper_rgb,
            self.sky_lower_rgb,
            self.sun_direction,
        )
        if any(len(value) != 3 for value in vectors):
            raise ValueError("Viewer vectors and colors must contain three values")
        scalars = (
            self.camera_pitch_deg,
            self.camera_yaw_deg,
            self.camera_fov_deg,
            self.camera_speed_m_s,
            self.shadow_radius_m,
            self.shadow_extents_m,
            self.diffuse_scale,
            self.specular_scale,
            self.exposure,
        )
        if not all(isfinite(value) for value in (*scalars, *(item for vector in vectors for item in vector))):
            raise ValueError("Viewer parameters must be finite")
        if not 0.0 < self.camera_fov_deg < 180.0:
            raise ValueError("camera_fov_deg must be between zero and 180")
        if self.camera_speed_m_s < 0.0 or self.shadow_radius_m < 0.0 or self.shadow_extents_m <= 0.0:
            raise ValueError("Viewer speed and shadow dimensions are invalid")
        if sum(value * value for value in self.sun_direction) == 0.0:
            raise ValueError("sun_direction must be nonzero")


VIEWER = ViewerConfig()
