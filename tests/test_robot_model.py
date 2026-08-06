"""Verify the ROS-aligned UR10e workcell model used by Newton."""

import xml.etree.ElementTree as ET

import newton
import numpy as np

from aic_newton.scene.assets import robot_description_path
from aic_newton.scene.robot import AIC_ARM_JOINT_NAMES, AIC_ROBOT_MODEL
from aic_newton.simulation.model import build_robot_model, build_simulation_model


def test_default_robot_description_is_ur10e_with_canonical_chain() -> None:
    path = robot_description_path()
    root = ET.parse(path).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    assert AIC_ROBOT_MODEL == "ur10e"
    assert tuple(name for name in AIC_ARM_JOINT_NAMES if name in joints) == AIC_ARM_JOINT_NAMES
    assert joints["elbow_joint"].find("origin").attrib["xyz"] == "-0.6127 0 0"
    assert joints["wrist_1_joint"].find("origin").attrib["xyz"] == "-0.57155 0 0.17415"
    assert joints["camera_mount_from_tool0_joint"].find("origin").attrib["xyz"] == "0.0 0.0 0.0"
    assert joints["hande_tcp_joint"].find("parent").attrib["link"] == "hande_base_link"
    assert joints["camera_center_optical_joint"].find("child").attrib["link"] == "camera_center_optical"


def test_newton_import_exposes_ros_aligned_ur10e_state() -> None:
    robot = build_robot_model()

    assert robot.model_name == "ur10e"
    assert robot.arm_joint_names == AIC_ARM_JOINT_NAMES
    assert robot.tool_body_label.endswith("hande_base_link")
    assert tuple(label.rsplit("/", 1)[-1] for label in robot.camera_body_labels) == (
        "camera_left_optical",
        "camera_center_optical",
        "camera_right_optical",
    )
    np.testing.assert_allclose(robot.joint_lower, (-2 * np.pi,) * 2 + (-np.pi,) + (-2 * np.pi,) * 3)
    np.testing.assert_allclose(robot.joint_upper, (2 * np.pi,) * 2 + (np.pi,) + (2 * np.pi,) * 3)


def test_simulation_keeps_render_only_visuals_for_robot_and_tool() -> None:
    """Catch hiding the URDF visuals together with its collision proxies."""
    model = build_simulation_model().model
    flags = model.shape_flags.numpy()
    shape_bodies = model.shape_body.numpy()
    shape_types = model.shape_type.numpy()

    for body_suffix in (
        "wrist_3_link",
        "ati_body",
        "hande_base_link",
        "hande_left_finger",
        "hande_right_finger",
        "camera_center",
    ):
        body = next(
            index for index, label in enumerate(model.body_label) if label.endswith(body_suffix)
        )
        visible_shapes = [
            shape
            for shape, shape_body in enumerate(shape_bodies)
            if shape_body == body and flags[shape] & int(newton.ShapeFlags.VISIBLE)
        ]
        assert visible_shapes, f"{body_suffix} has no camera-visible geometry"
        assert all(shape_types[shape] == int(newton.GeoType.MESH) for shape in visible_shapes)
        assert all(
            not flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES)
            for shape in visible_shapes
        )
