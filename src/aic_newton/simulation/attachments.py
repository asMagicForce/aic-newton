"""Manage fixed-joint ownership for cable connectors."""

from dataclasses import dataclass
from enum import IntEnum

import newton
import warp as wp
from newton.solvers import SolverVBD
from newton.solvers.experimental.coupled import SolverCoupledProxy

from ..config import CABLE
from ..utils.labels import find_label_index as _find_label_index
from ..utils.transforms import transform_from_row as _transform_from_row


class AttachmentMode(IntEnum):
    """Identify the active fixed-joint owner of the SFP module."""

    MOUNTED = 0
    GRASPED = 1
    SEATED = 2
    FAILED = 3


@dataclass(frozen=True)
class DynamicCableHandles:
    """Identify the dynamic AIC cable and connector bodies."""

    sfp_body: int
    sc_body: int
    cable_bodies: tuple[int, ...]
    cable_half_lengths: tuple[float, ...]
    sfp_root_joint: int
    mount_anchor_body: int
    seat_anchor_body: int
    sc_anchor_body: int
    mount_joint: int
    grasp_joint: int
    seat_joint: int
    sc_mount_joint: int
    bodies: tuple[int, ...]
    joints: tuple[int, ...]
    sfp_body_to_module: wp.transform


def _add_vbd_grasp_joint(
    builder: newton.ModelBuilder,
    *,
    tool_body: int,
    sfp_body: int,
    label: str,
) -> int:
    """Allocate the initially disabled fixed joint owned by the gripper."""
    tool_pose = builder.body_q[tool_body]
    sfp_pose = builder.body_q[sfp_body]
    return builder.add_joint_fixed(
        parent=tool_body,
        child=sfp_body,
        parent_xform=wp.transform_inverse(tool_pose) * sfp_pose,
        child_xform=wp.transform_identity(),
        collision_filter_parent=False,
        enabled=False,
        label=label,
    )


class VBDAttachmentOwnershipController:
    """Switch one dynamic cable endpoint between fixed VBD owners."""

    def __init__(
        self,
        *,
        solver: SolverCoupledProxy,
        free_root_joint_label: str,
        mount_joint_label: str,
        grasp_joint_label: str,
        seat_joint_label: str,
        sc_mount_joint_label: str,
    ):
        """Resolve fixed attachment joints and initialize mounted ownership."""
        self.view = solver.view("vbd")
        self.device = getattr(self.view, "device", self.view.joint_enabled.device)
        self.joint_enabled = self.view.joint_enabled
        self.joint_X_p = self.view.joint_X_p
        self.joint_X_c = self.view.joint_X_c
        self.body_q_rest = self.view.body_q
        self.free_root_joint = _find_label_index(self.view.joint_label, free_root_joint_label)
        self.mount_joint = _find_label_index(self.view.joint_label, mount_joint_label)
        self.grasp_joint = _find_label_index(self.view.joint_label, grasp_joint_label)
        self.seat_joint = _find_label_index(self.view.joint_label, seat_joint_label)
        self.sc_mount_joint = _find_label_index(self.view.joint_label, sc_mount_joint_label)

        joint_parent = self.view.joint_parent.numpy()
        joint_child = self.view.joint_child.numpy()
        self.tool_body = int(joint_parent[self.grasp_joint])
        self.sfp_body = int(joint_child[self.grasp_joint])
        self.mount_anchor_body = int(joint_parent[self.mount_joint])
        self.seat_anchor_body = int(joint_parent[self.seat_joint])
        self._original_tool_rest = _transform_from_row(self.body_q_rest.numpy()[self.tool_body])
        self._sfp_rest = _transform_from_row(self.body_q_rest.numpy()[self.sfp_body])
        self._bool_stage = wp.array([False], dtype=wp.bool, device=self.device)
        self._transform_stage = wp.array([wp.transform_identity()], dtype=wp.transform, device=self.device)
        self._penalty_stage = wp.array(
            [CABLE.attachment_stiffness, CABLE.attachment_stiffness],
            dtype=float,
            device=self.device,
        )
        self._configure_attachment_penalties(solver.solver("vbd"))
        self._align_anchor_rest_orientation(self.mount_anchor_body)
        self._align_anchor_rest_orientation(self.seat_anchor_body)
        self.mode = AttachmentMode.MOUNTED
        self._set_owner(self.mount_joint)

    def _configure_attachment_penalties(self, vbd_solver: SolverVBD) -> None:
        """Strengthen only fixed ownership slots relative to cable stretch."""
        constraint_starts = vbd_solver.joint_constraint_start.numpy()
        for joint in (self.mount_joint, self.grasp_joint, self.seat_joint, self.sc_mount_joint):
            start = int(constraint_starts[joint])
            for penalties in (
                vbd_solver.joint_penalty_k,
                vbd_solver.joint_penalty_k_min,
                vbd_solver.joint_penalty_k_max,
            ):
                wp.copy(penalties, self._penalty_stage, dest_offset=start, count=2)

    def _copy_enabled(self, joint: int, value: bool) -> None:
        """Copy one enabled flag without reallocating graph-visible arrays."""
        self._bool_stage.fill_(value)
        wp.copy(self.joint_enabled, self._bool_stage, dest_offset=joint, count=1)

    def _copy_transform(self, target: wp.array[wp.transform], index: int, value: wp.transform) -> None:
        """Copy one transform without reallocating graph-visible arrays."""
        self._transform_stage.fill_(value)
        wp.copy(target, self._transform_stage, dest_offset=index, count=1)

    def _align_anchor_rest_orientation(self, anchor_body: int) -> None:
        """Make a static anchor target the SFP's absolute current orientation."""
        anchor_rest = _transform_from_row(self.body_q_rest.numpy()[anchor_body])
        aligned_rest = wp.transform(
            wp.transform_get_translation(anchor_rest),
            wp.transform_get_rotation(self._sfp_rest),
        )
        self._copy_transform(self.body_q_rest, anchor_body, aligned_rest)

    def _set_owner(self, owner_joint: int) -> None:
        """Enable exactly one SFP fixed joint and retain the SC mount."""
        self._copy_enabled(self.free_root_joint, False)
        for joint in (self.mount_joint, self.grasp_joint, self.seat_joint):
            self._copy_enabled(joint, joint == owner_joint)
        self._copy_enabled(self.sc_mount_joint, True)

    def set_mode(
        self,
        mode: AttachmentMode,
        *,
        tool_pose: wp.transform | None = None,
        sfp_pose: wp.transform | None = None,
    ) -> None:
        """Atomically route the SFP fixed attachment to one owner."""
        if mode is AttachmentMode.FAILED:
            return
        if mode is AttachmentMode.GRASPED:
            if self.mode is not AttachmentMode.GRASPED:
                if tool_pose is None or sfp_pose is None:
                    raise ValueError("Entering GRASPED requires current tool and SFP poses")
                tool_to_sfp = wp.transform_inverse(tool_pose) * sfp_pose
                desired_rotation = wp.normalize(wp.transform_get_rotation(tool_to_sfp))
                rebased_tool_rest = wp.transform(
                    wp.transform_get_translation(self._original_tool_rest),
                    wp.normalize(wp.transform_get_rotation(self._sfp_rest) * wp.quat_inverse(desired_rotation)),
                )
                self._copy_transform(self.joint_X_p, self.grasp_joint, tool_to_sfp)
                self._copy_transform(self.joint_X_c, self.grasp_joint, wp.transform_identity())
                self._copy_transform(self.body_q_rest, self.tool_body, rebased_tool_rest)
            self._set_owner(self.grasp_joint)
        elif mode is AttachmentMode.MOUNTED:
            self._copy_transform(self.body_q_rest, self.tool_body, self._original_tool_rest)
            self._set_owner(self.mount_joint)
        elif mode is AttachmentMode.SEATED:
            self._copy_transform(self.body_q_rest, self.tool_body, self._original_tool_rest)
            self._set_owner(self.seat_joint)
        else:
            raise ValueError(f"Unsupported attachment mode: {mode}")
        self.mode = mode
