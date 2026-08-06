"""Stable headless control surface for the AIC Newton simulation."""

from __future__ import annotations

import threading
from copy import copy
from dataclasses import dataclass
from enum import Enum
from math import ceil, isfinite, sqrt
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp

from .config import CABLE, ROBOT_CONTROL, TASK_TARGET, TaskTargetConfig
from .scene.layout import component_placements, default_layout
from .scene.robot import AIC_ARM_JOINT_NAMES, AIC_GRIPPER_JOINT_NAMES, AIC_TOOL_TO_GRIPPER_TCP
from .simulation.application import Example
from .simulation.attachments import AttachmentMode
from .simulation.cameras import CameraSensor
from .types import CableSegmentSnapshot, CameraFrameSet, ObservationBundle, SceneSnapshot, StateSnapshot
from .utils.labels import find_label_index
from .utils.transforms import normalize_quaternion, pose_from_transform, rpy_quaternion, transform_from_row


class EngineLifecycle(Enum):
    CONFIGURED = "configured"
    RUNNING = "running"
    STOPPED = "stopped"
    CLOSED = "closed"


@dataclass(frozen=True)
class EngineConfig:
    """Construction-time options for deterministic headless execution."""

    substeps: int = 8
    cameras: bool = True
    graph_capture: bool = True
    camera_width: int = 288
    camera_height: int = 256
    camera_rate_hz: int = 20
    state_sample_rate_hz: int = 20
    task_target: TaskTargetConfig = TASK_TARGET

    def __post_init__(self) -> None:
        if self.substeps <= 0:
            raise ValueError("substeps must be positive")
        if self.camera_width <= 0 or self.camera_height <= 0 or self.camera_rate_hz <= 0:
            raise ValueError("camera dimensions and rate must be positive")
        if 60 % self.camera_rate_hz:
            raise ValueError("camera_rate_hz must divide the 60 Hz engine frame rate")
        if self.state_sample_rate_hz <= 0 or 60 % self.state_sample_rate_hz:
            raise ValueError("state_sample_rate_hz must divide the 60 Hz engine frame rate")


@dataclass(frozen=True)
class _ObservationRequest:
    state_index: int
    clock_time_s: float
    scene_time_s: float
    capture_state: bool
    render_camera: bool
    generation: int


class AICNewtonEngine:
    """Own one Newton scene without enabling demo autonomy or keyboard input."""

    def __init__(self, config: EngineConfig = EngineConfig()):
        self.config = config
        args = SimpleNamespace(
            auto=False,
            camera=False,
            substeps=config.substeps,
            cable_index=config.task_target.cable_index,
            nic_card_index=config.task_target.nic_card_index,
            nic_port_index=config.task_target.nic_port_index,
            graph_capture=config.graph_capture,
            camera_speed=0.0,
        )
        self._viewer = newton.viewer.ViewerNull(num_frames=1)
        self._example = Example(self._viewer, args)
        self._clock_time_s = 0.0
        self._frame_index = 0
        self._lifecycle = EngineLifecycle.CONFIGURED
        self._arm_q = self._coordinate_indices(AIC_ARM_JOINT_NAMES)
        self._arm_qd = self._dof_indices(AIC_ARM_JOINT_NAMES)
        self._gripper_q = self._coordinate_indices(AIC_GRIPPER_JOINT_NAMES)
        self._camera_sensor = (
            CameraSensor(
                copy(self._example.model),
                simulation_rate=self._example.fps,
                width=config.camera_width,
                height=config.camera_height,
                camera_rate=config.camera_rate_hz,
                initial_state=self._example.state_0,
            )
            if config.cameras
            else None
        )
        self._observation_states = tuple(self._example.model.state() for _ in range(2))
        self._observation_condition = threading.Condition()
        self._observation_free_states = set(range(len(self._observation_states)))
        self._observation_pending: _ObservationRequest | None = None
        self._observation_inflight: int | None = None
        self._observation_generation = 0
        self._observation_shutdown = False
        self._observation_thread: threading.Thread | None = None
        self._observation_error: Exception | None = None
        self._camera_error: Exception | None = None
        self._camera_frames: CameraFrameSet | None = None
        self._observation_snapshot = ObservationBundle(
            *self._read_snapshots(
                self._example.state_0,
                clock_time_s=self._clock_time_s,
                scene_time_s=self._example.sim_time,
            )
        )
        self._attachment_lock = threading.Lock()
        self._pending_attachment_modes: list[AttachmentMode] = []
        # Formal headless startup holds the configured joint state. The
        # standalone demo remains free to use its Cartesian keyboard target.
        self._write_joint_targets(self._observation_snapshot.state.joint_position_rad)

    @property
    def lifecycle(self) -> EngineLifecycle:
        return self._lifecycle

    @property
    def automatic_control_enabled(self) -> bool:
        return self._example.auto_enabled

    @property
    def keyboard_control_enabled(self) -> bool:
        return False

    @property
    def frame_dt_s(self) -> float:
        return self._example.frame_dt

    def _coordinate_indices(self, names: tuple[str, ...]) -> tuple[int, ...]:
        starts = self._example.model.joint_q_start.numpy()
        return tuple(int(starts[find_label_index(self._example.model.joint_label, name)]) for name in names)

    def _dof_indices(self, names: tuple[str, ...]) -> tuple[int, ...]:
        starts = self._example.model.joint_qd_start.numpy()
        return tuple(int(starts[find_label_index(self._example.model.joint_label, name)]) for name in names)

    def start(self) -> None:
        if self._lifecycle is EngineLifecycle.CLOSED:
            raise RuntimeError("engine is closed")
        self._lifecycle = EngineLifecycle.RUNNING
        if self._observation_thread is None:
            self._observation_thread = threading.Thread(
                target=self._observation_loop,
                name="aic-newton-observation",
                daemon=True,
            )
            self._observation_thread.start()

    def _queue_observation(self, *, capture_state: bool, render_camera: bool) -> None:
        """Copy the latest GPU state into a free latest-only observation slot."""
        with self._observation_condition:
            if not self._observation_free_states and self._observation_pending is not None:
                capture_state |= self._observation_pending.capture_state
                render_camera |= self._observation_pending.render_camera
                self._observation_free_states.add(self._observation_pending.state_index)
                self._observation_pending = None
            if not self._observation_free_states:
                return
            state_index = self._observation_free_states.pop()
            self._observation_states[state_index].assign(self._example.state_0)
            self._observation_pending = _ObservationRequest(
                state_index=state_index,
                clock_time_s=self._clock_time_s,
                scene_time_s=self._example.sim_time,
                capture_state=capture_state,
                render_camera=render_camera,
                generation=self._observation_generation,
            )
            self._observation_condition.notify()

    def _observation_loop(self) -> None:
        """Read state and render only the newest sample off the physics thread."""
        while True:
            with self._observation_condition:
                self._observation_condition.wait_for(
                    lambda: self._observation_shutdown or self._observation_pending is not None
                )
                if self._observation_shutdown:
                    return
                assert self._observation_pending is not None
                request = self._observation_pending
                self._observation_pending = None
                self._observation_inflight = request.state_index
                sensor = self._camera_sensor
            try:
                state = self._observation_states[request.state_index]
                snapshots = None
                frames = None
                state_error = None
                camera_error = None
                if request.capture_state:
                    try:
                        snapshots = self._read_snapshots(
                            state,
                            clock_time_s=request.clock_time_s,
                            scene_time_s=request.scene_time_s,
                        )
                    except Exception as error:
                        state_error = error
                if request.render_camera:
                    try:
                        assert sensor is not None
                        images = sensor.render_now(state)
                        _, height, width, _ = images.shape
                        frames = CameraFrameSet(
                            clock_time_s=request.clock_time_s,
                            width=width,
                            height=height,
                            left_rgb=images[0].tobytes(),
                            center_rgb=images[1].tobytes(),
                            right_rgb=images[2].tobytes(),
                        )
                    except Exception as error:
                        camera_error = error
                with self._observation_condition:
                    if request.generation == self._observation_generation:
                        if snapshots is not None:
                            self._observation_snapshot = ObservationBundle(*snapshots)
                        if frames is not None:
                            self._camera_frames = frames
                    if request.capture_state:
                        self._observation_error = state_error
                    if request.render_camera:
                        self._camera_error = camera_error
            finally:
                with self._observation_condition:
                    self._observation_inflight = None
                    self._observation_free_states.add(request.state_index)

    def step(self, frames: int = 1) -> None:
        if self._lifecycle is not EngineLifecycle.RUNNING:
            raise RuntimeError("engine must be running before step")
        if frames <= 0:
            raise ValueError("frames must be positive")
        for _ in range(frames):
            self._apply_pending_attachment_modes()
            if self._example.graph is None:
                self._example.simulate()
            else:
                wp.capture_launch(self._example.graph)
            self._example.sim_time += self._example.frame_dt
            self._clock_time_s += self._example.frame_dt
            self._frame_index += 1
            state_due = self._frame_index % (60 // self.config.state_sample_rate_hz) == 0
            camera_due = self._camera_sensor is not None and self._frame_index % (60 // self.config.camera_rate_hz) == 0
            if state_due or camera_due:
                self._queue_observation(
                    capture_state=state_due,
                    render_camera=camera_due,
                )

    def snapshot(self) -> StateSnapshot:
        """Return the immutable snapshot captured after the latest physics frame."""
        if self._observation_error is not None:
            raise RuntimeError(f"state observation failed: {self._observation_error}") from self._observation_error
        return self._observation_snapshot.state

    def observation_snapshot(self) -> ObservationBundle:
        """Return robot and scene values captured from exactly one physics frame."""
        if self._observation_error is not None:
            raise RuntimeError(f"state observation failed: {self._observation_error}") from self._observation_error
        return self._observation_snapshot

    def _read_snapshots(
        self,
        state: newton.State,
        *,
        clock_time_s: float,
        scene_time_s: float,
    ) -> tuple[StateSnapshot, SceneSnapshot]:
        """Read one coherent GPU state into immutable public values."""
        joint_q = state.joint_q.numpy()
        joint_qd = state.joint_qd.numpy()
        body_q = state.body_q.numpy()
        tool_pose = pose_from_transform(transform_from_row(body_q[self._example.tool_body]) * AIC_TOOL_TO_GRIPPER_TCP)
        state_snapshot = StateSnapshot(
            clock_time_s=clock_time_s,
            scene_time_s=scene_time_s,
            joint_names=AIC_ARM_JOINT_NAMES,
            joint_position_rad=tuple(float(joint_q[index]) for index in self._arm_q),
            joint_velocity_rad_s=tuple(float(joint_qd[index]) for index in self._arm_qd),
            gripper_position_m=tuple(float(joint_q[index]) for index in self._gripper_q),
            tcp_pose_xyz_xyzw=(*tool_pose.xyz, *tool_pose.quat_xyzw),
            tcp_twist_linear_angular=(0.0,) * 6,
            tcp_wrench_force_torque=(0.0,) * 6,
        )
        handles = self._example.dynamic_cable
        layout = default_layout()
        board_quaternion = rpy_quaternion(layout.board_rpy)
        sfp_body_pose = transform_from_row(body_q[handles.sfp_body])
        sfp_module_pose = pose_from_transform(sfp_body_pose * handles.sfp_body_to_module)
        target_pose = self._example.manipulation_frames.port_bottom
        cable_segments = tuple(
            CableSegmentSnapshot(
                pose_xyz_xyzw=(*pose.xyz, *pose.quat_xyzw),
                half_length_m=half_length,
                radius_m=CABLE.radius_m,
            )
            for index, half_length in zip(handles.cable_bodies, handles.cable_half_lengths, strict=True)
            for pose in (pose_from_transform(transform_from_row(body_q[index])),)
        )
        scene_snapshot = SceneSnapshot(
            clock_time_s=clock_time_s,
            board_pose_xyz_xyzw=(*layout.board_xyz, *board_quaternion),
            components=component_placements(layout),
            static_cables=tuple(
                cable
                for cable in self._example.scene.cable_assemblies
                if cable.name != self._example.manipulation_frames.cable_name
            ),
            manipulation_frames=self._example.manipulation_frames,
            cable_points_xyz=tuple(segment.pose_xyz_xyzw[:3] for segment in cable_segments),
            cable_segments=cable_segments,
            manipulated_object_pose_xyz_xyzw=(*sfp_module_pose.xyz, *sfp_module_pose.quat_xyzw),
            target_pose_xyz_xyzw=(*target_pose.xyz, *target_pose.quat_xyzw),
        )
        return state_snapshot, scene_snapshot

    def reset(self) -> None:
        if self._lifecycle is EngineLifecycle.CLOSED:
            raise RuntimeError("engine is closed")
        with self._observation_condition:
            self._observation_generation += 1
            if self._observation_pending is not None:
                self._observation_free_states.add(self._observation_pending.state_index)
                self._observation_pending = None
        with self._attachment_lock:
            self._pending_attachment_modes.clear()
            self._example.reset()
        with self._observation_condition:
            if self._camera_sensor is not None:
                self._camera_sensor.reset()
            self._camera_frames = None
            self._camera_error = None
            self._observation_error = None
        self._observation_snapshot = ObservationBundle(
            *self._read_snapshots(
                self._example.state_0,
                clock_time_s=self._clock_time_s,
                scene_time_s=self._example.sim_time,
            )
        )

    def camera_snapshot(self) -> CameraFrameSet | None:
        """Return the most recent coherent three-view sample without blocking."""
        if self._camera_error is not None:
            raise RuntimeError(f"camera render failed: {self._camera_error}") from self._camera_error
        return self._camera_frames

    def scene_snapshot(self) -> SceneSnapshot:
        """Return task geometry from the same Newton state as robot feedback."""
        return self._observation_snapshot.scene

    @staticmethod
    def _validate_pose(target: tuple[float, ...]) -> tuple[float, ...]:
        if len(target) != 7:
            raise ValueError("TCP target must contain seven xyz+xyzw values")
        if not all(isfinite(value) for value in target):
            raise ValueError("TCP target must contain finite values")
        quaternion = normalize_quaternion(tuple(target[3:]), field="TCP target quaternion")
        return (*target[:3], *quaternion)

    def _require_running(self) -> None:
        if self._lifecycle is not EngineLifecycle.RUNNING:
            raise RuntimeError("engine must be running")

    def _write_joint_targets(self, target: tuple[float, ...]) -> None:
        values = self._example.control.joint_target_q.numpy()
        velocities = self._example.control.joint_target_qd.numpy()
        for index, value in zip(self._arm_q, target, strict=True):
            values[index] = value
        for index in self._arm_qd:
            velocities[index] = 0.0
        self._example.control.joint_target_q.assign(values)
        self._example.control.joint_target_qd.assign(velocities)
        self._example.external_joint_control = True

    def set_joint_target(self, target: tuple[float, ...]) -> None:
        """Replace the current joint target without advancing simulation time."""
        self._require_running()
        if len(target) != 6:
            raise ValueError("joint target must contain six values")
        if not all(isfinite(value) for value in target):
            raise ValueError("joint target must contain finite values")
        self._write_joint_targets(target)

    def move_j(self, target: tuple[float, ...], *, duration_s: float = 1.0) -> None:
        """Execute one time-parameterized joint-space move."""
        self._require_running()
        if len(target) != 6:
            raise ValueError("MoveJ target must contain six joint values")
        if not all(isfinite(value) for value in target):
            raise ValueError("MoveJ target must contain finite values")
        if duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        start = self.snapshot().joint_position_rad
        frames = max(1, ceil(duration_s / self.frame_dt_s))
        for frame in range(1, frames + 1):
            ratio = frame / frames
            command = tuple(a + ratio * (b - a) for a, b in zip(start, target, strict=True))
            self.set_joint_target(command)
            self.step()

    def _set_tcp_target(self, target: tuple[float, ...]) -> None:
        validated = self._validate_pose(target)
        self._example.external_joint_control = False
        self._example.tcp_controller.set_target(wp.transform(wp.vec3(*validated[:3]), wp.quat(*validated[3:])))

    def set_tcp_target(self, target: tuple[float, ...]) -> None:
        """Replace the current Cartesian target without advancing simulation time."""
        self._require_running()
        self._set_tcp_target(target)

    def move_l(self, target: tuple[float, ...], *, duration_s: float = 1.0) -> None:
        """Execute one time-parameterized Cartesian TCP move."""
        self._require_running()
        target = self._validate_pose(target)
        if duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        start = self.snapshot().tcp_pose_xyz_xyzw
        start_q = np.asarray(start[3:], dtype=float)
        target_q = np.asarray(target[3:], dtype=float)
        if float(np.dot(start_q, target_q)) < 0.0:
            target_q = -target_q
        frames = max(1, ceil(duration_s / self.frame_dt_s))
        for frame in range(1, frames + 1):
            ratio = frame / frames
            xyz = tuple(a + ratio * (b - a) for a, b in zip(start[:3], target[:3], strict=True))
            quaternion = (1.0 - ratio) * start_q + ratio * target_q
            norm = sqrt(float(np.dot(quaternion, quaternion)))
            self.set_tcp_target((*xyz, *(quaternion / norm)))
            self.step()

    def set_servo_l_target(self, target: tuple[float, ...]) -> None:
        """Replace the current Cartesian servo target without queuing history."""
        self._require_running()
        self.set_tcp_target(target)

    def set_gripper(self, normalized_position: float) -> None:
        """Set normalized Hand-E closure, where zero is open and one is closed."""
        self._require_running()
        if not isfinite(normalized_position) or not 0.0 <= normalized_position <= 1.0:
            raise ValueError("normalized_position must be between zero and one")
        position = ROBOT_CONTROL.gripper_open_q + normalized_position * (
            ROBOT_CONTROL.gripper_closed_q - ROBOT_CONTROL.gripper_open_q
        )
        values = self._example.control.joint_target_q.numpy()
        for index in self._gripper_q:
            values[index] = position
        self._example.control.joint_target_q.assign(values)

    @property
    def attachment_mode(self) -> str:
        """Return simulation-only ownership for the manipulated connector."""
        with self._attachment_lock:
            return self._example.vbd_attachment_controller.mode.name.lower()

    def set_attachment_mode(self, mode: str) -> None:
        """Queue physical grasp/seat ownership for the next Newton tick."""
        self._require_running()
        try:
            requested = AttachmentMode[mode.strip().upper()]
        except (AttributeError, KeyError) as error:
            raise ValueError(f"invalid attachment mode: {mode!r}") from error
        with self._attachment_lock:
            self._pending_attachment_modes.append(requested)

    def _apply_pending_attachment_modes(self) -> None:
        """Apply queued ownership changes while the physics thread owns solver state."""
        with self._attachment_lock:
            if not self._pending_attachment_modes:
                return
            controller = self._example.vbd_attachment_controller
            for requested in self._pending_attachment_modes:
                if requested is AttachmentMode.GRASPED and controller.mode is not requested:
                    body_q = self._example.state_0.body_q.numpy()
                    controller.set_mode(
                        requested,
                        tool_pose=transform_from_row(body_q[self._example.tool_body]),
                        sfp_pose=transform_from_row(body_q[self._example.dynamic_cable.sfp_body]),
                    )
                else:
                    controller.set_mode(requested)
            self._pending_attachment_modes.clear()

    def stop_motion(self) -> None:
        """Discard any Cartesian trajectory and hold the current arm position."""
        self._require_running()
        self._write_joint_targets(self.snapshot().joint_position_rad)

    def stop(self) -> None:
        if self._lifecycle is EngineLifecycle.CLOSED:
            return
        self._lifecycle = EngineLifecycle.STOPPED
        with self._observation_condition:
            self._observation_generation += 1
            if self._observation_pending is not None:
                self._observation_free_states.add(self._observation_pending.state_index)
                self._observation_pending = None
            self._observation_condition.notify_all()

    def close(self) -> None:
        if self._lifecycle is EngineLifecycle.CLOSED:
            return
        self._lifecycle = EngineLifecycle.CLOSED
        with self._observation_condition:
            self._observation_shutdown = True
            self._observation_condition.notify_all()
        if self._observation_thread is not None:
            self._observation_thread.join(timeout=5.0)
            if self._observation_thread.is_alive():
                raise RuntimeError("observation worker did not stop")
        close = getattr(self._viewer, "close", None)
        if close is not None:
            close()


__all__ = [
    "AICNewtonEngine",
    "EngineConfig",
    "EngineLifecycle",
    "SceneSnapshot",
    "StateSnapshot",
]
