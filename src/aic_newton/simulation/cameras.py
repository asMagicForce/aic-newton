"""Configure and render the three task-board cameras."""

import argparse

import newton
import numpy as np
import warp as wp
from newton.sensors import SensorTiledCamera

from ..config import CAMERA
from ..utils.labels import find_label_index

CAMERA_BODY_SUFFIXES = (
    "camera_left_optical",
    "camera_center_optical",
    "camera_right_optical",
)
CAMERA_OPTICAL_TO_NEWTON = wp.transform(
    wp.vec3(0.0),
    wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.pi)),
)


def vertical_fov(horizontal_fov: float, width: int, height: int) -> float:
    """Convert a horizontal pinhole FOV to its vertical FOV."""
    return 2.0 * float(np.arctan(np.tan(horizontal_fov / 2.0) * height / width))


def camera_update_due(frame: int, *, simulation_rate: int, camera_rate: int) -> bool:
    """Return whether a camera update is due on a simulation frame."""
    if simulation_rate % camera_rate != 0:
        raise ValueError("Camera update rate must divide the simulation rate")
    return frame % (simulation_rate // camera_rate) == 0


def nonnegative_finite_float(value: str) -> float:
    """Parse a finite nonnegative command-line float."""
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return result


def camera_render_config() -> SensorTiledCamera.RenderConfig:
    """Configure clean textured output for duplicated double-sided meshes."""
    return SensorTiledCamera.RenderConfig(
        enable_textures=True,
        enable_shadows=False,
        enable_backface_culling=False,
        render_order=SensorTiledCamera.RenderOrder.TILED,
        max_distance=CAMERA.far_clip_m - CAMERA.near_clip_m,
    )


def log_camera_images(viewer, images) -> None:
    """Display the left, center, and right cameras as one vertical strip."""
    camera_count, height, width, channels = images.shape
    viewer.log_image(CAMERA.panel_name, images.reshape((camera_count * height, width, channels)))


def camera_window_geometry(
    display_width: float,
    display_height: float,
    *,
    ui_scale: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Size and place the camera strip at the viewport's bottom-right."""
    margin = CAMERA.panel_margin_px * ui_scale
    available_width = max(1.0, display_width - 2.0 * margin)
    available_height = max(1.0, display_height - 2.0 * margin)
    width = min(CAMERA.panel_width_px * ui_scale, available_width)
    image_height = width * len(CAMERA_BODY_SUFFIXES) * CAMERA.height_px / CAMERA.width_px
    height = min(image_height + CAMERA.panel_title_height_px * ui_scale, available_height)
    return (
        (display_width - width - margin, display_height - height - margin),
        (width, height),
    )


def position_camera_window(imgui) -> None:
    """Keep the Viewer image window at the bottom-right."""
    display_size = imgui.get_io().display_size
    ui_scale = max(1.0, float(imgui.get_font_size()) / 13.0)
    position, size = camera_window_geometry(
        float(display_size.x),
        float(display_size.y),
        ui_scale=ui_scale,
    )
    imgui.set_window_size(CAMERA.panel_name, imgui.ImVec2(*size))
    imgui.set_window_pos(CAMERA.panel_name, imgui.ImVec2(*position))


class CameraSensor:
    """Render three synchronized wrist-camera images independently of a Viewer."""

    def __init__(
        self,
        model: newton.Model,
        *,
        simulation_rate: int,
        width: int = CAMERA.width_px,
        height: int = CAMERA.height_px,
        camera_rate: int = CAMERA.update_rate_hz,
        viewer=None,
        initial_state: newton.State | None = None,
    ):
        self.model = model
        self.viewer = viewer
        self.simulation_rate = simulation_rate
        self.width = width
        self.height = height
        self.camera_rate = camera_rate
        self.frame = 0
        if width <= 0 or height <= 0 or camera_rate <= 0:
            raise ValueError("camera dimensions and rate must be positive")
        if simulation_rate % camera_rate:
            raise ValueError("camera rate must divide the simulation rate")
        if initial_state is not None:
            self.model.bvh_build_shapes(initial_state)
            self.model.bvh_build_particles(initial_state)

        camera_count = len(CAMERA_BODY_SUFFIXES)
        camera_bodies = [find_label_index(model.body_label, suffix) for suffix in CAMERA_BODY_SUFFIXES]
        self.body_indices = wp.array(camera_bodies, dtype=int, device=model.device)
        self.transforms = wp.zeros(
            (camera_count, 1),
            dtype=wp.transform,
            device=model.device,
        )

        self.sensor = SensorTiledCamera(
            model=model,
            default_render_config=camera_render_config(),
        )
        self.sensor.utils.create_default_light(enable_shadows=False)
        fov = vertical_fov(CAMERA.horizontal_fov_rad, width, height)
        self.rays = self.sensor.utils.compute_camera_rays_pinhole(
            width,
            height,
            camera_fovs=[fov] * camera_count,
        )
        wp.launch(
            apply_camera_near_clip,
            dim=(camera_count, height, width),
            inputs=[CAMERA.near_clip_m, self.rays],
            device=model.device,
        )
        self.color_image = self.sensor.utils.create_color_image_output(
            width,
            height,
            camera_count,
        )
        self.rgba = self.sensor.utils.to_rgba_from_color(self.color_image)

    def reset(self) -> None:
        """Restart the sensor update cadence."""
        self.frame = 0

    def render(self, state: newton.State) -> np.ndarray | None:
        """Return one contiguous left/center/right RGB set when an update is due."""
        if not camera_update_due(
            self.frame,
            simulation_rate=self.simulation_rate,
            camera_rate=self.camera_rate,
        ):
            self.frame += 1
            return None

        images = self.render_now(state)
        self.frame += 1
        return images

    def render_now(self, state: newton.State) -> np.ndarray:
        """Render one coherent frame set without applying an internal cadence."""

        wp.launch(
            update_camera_transforms,
            dim=len(CAMERA_BODY_SUFFIXES),
            inputs=[
                state.body_q,
                self.body_indices,
                CAMERA_OPTICAL_TO_NEWTON,
                self.transforms,
            ],
            device=self.model.device,
        )
        self.model.bvh_refit_shapes(state)
        self.model.bvh_refit_particles(state)
        self.sensor.update(
            state,
            self.transforms,
            self.rays,
            color_image=self.color_image,
        )
        rgba = self.rgba.numpy()
        rgb = np.ascontiguousarray(rgba[..., :3])
        if np.issubdtype(rgb.dtype, np.floating):
            rgb = np.asarray(np.clip(rgb, 0.0, 1.0) * 255.0, dtype=np.uint8)
        elif rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        if self.viewer is not None:
            log_camera_images(self.viewer, rgba)
        return rgb


class CameraPanel(CameraSensor):
    """Backward-compatible Viewer client for the standalone demo."""

    def __init__(self, model: newton.Model, viewer, *, simulation_rate: int):
        super().__init__(model, simulation_rate=simulation_rate, viewer=viewer)


@wp.kernel
def update_camera_transforms(
    body_q: wp.array[wp.transform],
    body_indices: wp.array[int],
    optical_to_newton: wp.transform,
    camera_transforms: wp.array2d[wp.transform],
):
    """Update camera poses from their attached rigid bodies."""
    camera = wp.tid()
    camera_transforms[camera, 0] = wp.transform_multiply(body_q[body_indices[camera]], optical_to_newton)


@wp.kernel
def apply_camera_near_clip(
    near: float,
    camera_rays: wp.array4d[wp.vec3],
):
    """Move camera-ray origins to the configured near plane."""
    camera, y, x = wp.tid()
    camera_rays[camera, y, x, 0] = camera_rays[camera, y, x, 1] * near
