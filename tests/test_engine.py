"""Behavioral contract for the ROS-independent headless engine."""

import math
import threading
import time

import numpy as np
import pytest

from aic_newton.engine import AICNewtonEngine, EngineConfig, EngineLifecycle


@pytest.fixture(scope="module")
def engine() -> AICNewtonEngine:
    instance = AICNewtonEngine(
        EngineConfig(
            cameras=True,
            graph_capture=False,
            substeps=2,
            camera_width=96,
            camera_height=64,
            camera_rate_hz=20,
            state_sample_rate_hz=60,
        )
    )
    yield instance
    instance.close()


def test_headless_engine_starts_without_demo_control(engine: AICNewtonEngine) -> None:
    assert engine.lifecycle is EngineLifecycle.CONFIGURED
    assert not engine.automatic_control_enabled
    assert not engine.keyboard_control_enabled

    engine.start()

    assert engine.lifecycle is EngineLifecycle.RUNNING


def test_step_produces_immutable_canonical_state(engine: AICNewtonEngine) -> None:
    before = engine.snapshot()
    assert engine.snapshot() is before
    assert engine.scene_snapshot() is engine.scene_snapshot()
    engine.step(2)
    after = engine.snapshot()

    assert after is not before
    assert after.clock_time_s > before.clock_time_s
    assert after.scene_time_s > before.scene_time_s
    assert after.joint_names == (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    assert len(after.joint_position_rad) == len(after.joint_velocity_rad_s) == 6
    assert len(after.tcp_pose_xyz_xyzw) == 7
    with pytest.raises(Exception):
        after.joint_position_rad[0] = 0.0  # type: ignore[index]


def test_reset_restores_scene_without_rewinding_clock(engine: AICNewtonEngine) -> None:
    before = engine.snapshot()
    engine.reset()
    after = engine.snapshot()

    assert after.clock_time_s >= before.clock_time_s
    assert after.scene_time_s == 0.0


def test_camera_snapshot_is_one_synchronized_rgb_frame_set(engine: AICNewtonEngine) -> None:
    engine.step(3)
    frames = engine.camera_snapshot()

    assert frames is not None
    assert frames.clock_time_s <= engine.snapshot().clock_time_s
    assert (frames.width, frames.height) == (96, 64)
    assert len(frames.left_rgb) == len(frames.center_rgb) == len(frames.right_rgb)
    assert len(frames.left_rgb) == frames.width * frames.height * 3
    assert frames.left_rgb != frames.center_rgb


def test_slow_camera_render_does_not_block_physics_step(engine: AICNewtonEngine) -> None:
    """Catch camera rendering or GPU readback running inline with physics."""

    class BlockingCameraSensor:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()
            self.completed = threading.Event()

        def render(self, _state):
            self.entered.set()
            self.release.wait(timeout=1.0)
            self.completed.set()
            return None

        render_now = render

        def reset(self) -> None:
            pass

    original = engine._camera_sensor
    sensor = BlockingCameraSensor()
    engine._camera_sensor = sensor
    engine.start()
    try:
        started = time.monotonic()
        engine.step(3)
        elapsed = time.monotonic() - started

        assert sensor.entered.wait(timeout=0.2)
        assert elapsed < 0.5
    finally:
        sensor.release.set()
        if sensor.entered.is_set():
            assert sensor.completed.wait(timeout=1.0)
        deadline = time.monotonic() + 1.0
        while engine._camera_error is None and time.monotonic() < deadline:
            time.sleep(0.001)
        engine._camera_sensor = original
    assert engine.snapshot().clock_time_s > 0.0


def test_slow_state_readback_does_not_block_physics_step(engine: AICNewtonEngine) -> None:
    """Catch GPU-to-CPU observation reads running inline with physics."""
    entered = threading.Event()
    release = threading.Event()
    original = engine._read_snapshots

    def blocking_read(*args, **kwargs):
        entered.set()
        release.wait(timeout=1.0)
        return original(*args, **kwargs)

    engine._read_snapshots = blocking_read
    engine.start()
    before = engine.snapshot().clock_time_s
    try:
        started = time.monotonic()
        engine.step(3)
        elapsed = time.monotonic() - started

        assert entered.wait(timeout=0.2)
        assert elapsed < 0.5
    finally:
        release.set()
        deadline = time.monotonic() + 1.0
        while engine.snapshot().clock_time_s <= before and time.monotonic() < deadline:
            time.sleep(0.001)
        engine._read_snapshots = original
    assert engine.snapshot().clock_time_s > before


def test_scene_snapshot_exposes_complete_task_geometry(engine: AICNewtonEngine) -> None:
    scene = engine.scene_snapshot()

    assert scene.clock_time_s == engine.snapshot().clock_time_s
    assert len(scene.board_pose_xyz_xyzw) == 7
    assert {component.asset_name for component in scene.components} >= {
        "SFP Mount",
        "SC Mount",
        "SC Port",
        "NIC Card Mount",
        "NIC Card",
    }
    assert len(scene.static_cables) == 4
    assert all(cable.name != scene.manipulation_frames.cable_name for cable in scene.static_cables)
    assert len(scene.manipulation_frames.port_entrance.xyz) == 3
    assert len(scene.cable_points_xyz) > 2
    assert all(len(point) == 3 for point in scene.cable_points_xyz)
    assert len(scene.cable_segments) == len(scene.cable_points_xyz)
    assert all(
        len(segment.pose_xyz_xyzw) == 7
        and all(math.isfinite(value) for value in segment.pose_xyz_xyzw)
        and math.isclose(
            math.sqrt(sum(value * value for value in segment.pose_xyz_xyzw[3:])),
            1.0,
            abs_tol=1.0e-6,
        )
        and segment.half_length_m > 0.0
        and segment.radius_m == pytest.approx(0.002)
        for segment in scene.cable_segments
    )
    assert len(scene.manipulated_object_pose_xyz_xyzw) == 7
    assert len(scene.target_pose_xyz_xyzw) == 7


def test_observation_snapshot_keeps_robot_and_scene_on_one_clock(
    engine: AICNewtonEngine,
) -> None:
    """Catch callers combining robot and scene values from different physics frames."""
    observation = engine.observation_snapshot()

    assert observation.state.clock_time_s == observation.scene.clock_time_s


def test_camera_render_does_not_refit_the_physics_model_bvh(
    engine: AICNewtonEngine,
) -> None:
    """Catch the asynchronous camera worker mutating collision broad-phase state."""
    model = engine._example.model
    sensor = engine._camera_sensor
    assert sensor is not None
    probe = model.state()
    probe.assign(engine._example.state_0)
    body_q = probe.body_q.numpy()
    sfp_body = engine._example.dynamic_cable.sfp_body
    body_q[sfp_body, 0] += 0.5
    probe.body_q.assign(body_q)
    before = model.bvh_shapes.lowers.numpy().copy()

    try:
        sensor.render_now(probe)
        after = model.bvh_shapes.lowers.numpy()
        assert np.array_equal(after, before)
    finally:
        model.bvh_refit_shapes(engine._example.state_0)
        model.bvh_refit_particles(engine._example.state_0)


def test_engine_accepts_joint_tcp_servo_and_gripper_commands(engine: AICNewtonEngine) -> None:
    engine.start()
    initial = engine.snapshot()
    joint_target = list(initial.joint_position_rad)
    joint_target[0] += 0.03

    engine.move_j(tuple(joint_target), duration_s=0.2)
    moved = engine.snapshot()
    assert moved.joint_position_rad[0] > initial.joint_position_rad[0]

    tcp_target = moved.tcp_pose_xyz_xyzw
    engine.move_l(tcp_target, duration_s=0.1)
    engine.set_servo_l_target(tcp_target)
    engine.set_gripper(1.0)
    engine.step(2)
    engine.stop_motion()

    assert all(position > 0.0 for position in engine.snapshot().gripper_position_m)


def test_engine_exposes_simulation_attachment_adapter(engine: AICNewtonEngine) -> None:
    engine.start()
    engine.set_attachment_mode("mounted")
    assert engine.attachment_mode == "mounted"

    engine.set_attachment_mode("grasped")
    assert engine.attachment_mode == "mounted"
    engine.step()
    assert engine.attachment_mode == "grasped"

    with pytest.raises(ValueError, match="attachment mode"):
        engine.set_attachment_mode("invalid")


@pytest.mark.parametrize(
    ("method", "args", "message"),
    (
        ("move_j", ((0.0,) * 5,), "six"),
        ("move_l", ((0.0,) * 6,), "seven"),
        ("set_servo_l_target", ((0.0,) * 8,), "seven"),
        ("set_gripper", (-0.1,), "between"),
    ),
)
def test_engine_rejects_malformed_motion_commands(
    engine: AICNewtonEngine,
    method: str,
    args: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        getattr(engine, method)(*args)


def test_stop_restart_and_repeated_close_are_clean(engine: AICNewtonEngine) -> None:
    engine.stop()
    assert engine.lifecycle is EngineLifecycle.STOPPED
    with pytest.raises(RuntimeError, match="running"):
        engine.step()

    engine.start()
    engine.step()
    engine.close()
    engine.close()
    assert engine.lifecycle is EngineLifecycle.CLOSED
