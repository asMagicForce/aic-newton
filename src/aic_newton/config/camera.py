"""Configure task-board sensor cameras and their Viewer panel."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class CameraConfig:
    """Configure sensor rendering and image-panel presentation."""

    width_px: int = 288
    height_px: int = 256
    update_rate_hz: int = 20
    horizontal_fov_rad: float = 0.8718
    near_clip_m: float = 0.07
    far_clip_m: float = 20.0
    panel_name: str = "Basler cameras"
    panel_width_px: float = 320.0
    panel_margin_px: float = 10.0
    panel_title_height_px: float = 36.0

    def __post_init__(self) -> None:
        """Validate camera dimensions, rates, and clipping planes."""
        if self.width_px <= 0 or self.height_px <= 0 or self.update_rate_hz <= 0:
            raise ValueError("Camera dimensions and update rate must be positive")
        values = (
            self.horizontal_fov_rad,
            self.near_clip_m,
            self.far_clip_m,
            self.panel_width_px,
            self.panel_margin_px,
            self.panel_title_height_px,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Camera parameters must be finite")
        if not 0.0 < self.horizontal_fov_rad < 3.141592653589793:
            raise ValueError("horizontal_fov_rad must be between zero and pi")
        if self.near_clip_m < 0.0 or self.far_clip_m <= self.near_clip_m:
            raise ValueError("Camera clipping planes must satisfy 0 <= near < far")
        if self.panel_width_px <= 0.0 or self.panel_margin_px < 0.0 or self.panel_title_height_px < 0.0:
            raise ValueError("Camera panel dimensions must be nonnegative with positive width")
        if not self.panel_name:
            raise ValueError("panel_name must not be empty")


CAMERA = CameraConfig()
