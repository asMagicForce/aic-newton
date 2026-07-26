# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Control the automatic SFP extraction and insertion sequence."""

from dataclasses import dataclass
from enum import Enum, auto
from math import hypot, isfinite

from ..config import AUTO_INSERTION, ROBOT_CONTROL, TASK_TARGET, TaskTargetConfig
from ..scene.layout import ManipulationFrames
from ..simulation.attachments import AttachmentMode
from ..utils.transforms import (
    PoseTuple,
)
from ..utils.transforms import (
    compose_pose as _compose_pose,
)
from ..utils.transforms import (
    inverse_pose as _inverse_pose,
)
from ..utils.transforms import (
    normalize_direction as _normalize_direction,
)
from ..utils.transforms import (
    quat_rotate as _quat_rotate,
)
from ..utils.transforms import (
    quaternion_from_axes as _quaternion_from_axes,
)
from .trajectory import (
    MINIMUM_JERK_PEAK_SPEED_SCALE,
    interpolate_pose,
    minimum_jerk,
    orientation_error,
    translation_error,
    validate_motion_segment,
)


class AutoState(Enum):
    """Identify one step in the automatic SFP insertion sequence."""

    HOME = auto()
    MOVE_ABOVE_SFP = auto()
    DESCEND_TO_GRASP = auto()
    CLOSE_GRIPPER = auto()
    EXTRACT_FROM_MOUNT = auto()
    LIFT_AFTER_EXTRACTION = auto()
    TRANSFER_ABOVE_PORT = auto()
    ALIGN_WITH_PORT = auto()
    INSERT_TO_BOTTOM = auto()
    OPEN_GRIPPER = auto()
    RETRACT_FROM_PORT = auto()
    LIFT_AFTER_RELEASE = auto()
    COMPLETE = auto()
    FAILED = auto()


@dataclass(frozen=True)
class AutoObservation:
    """Describe the simulation state observed by the automatic controller."""

    tcp_pose: PoseTuple
    sfp_pose: PoseTuple
    arm_q: tuple[float, ...]
    gripper_q: tuple[float, float]
    finite: bool = True


@dataclass(frozen=True)
class AutoCommand:
    """Describe the TCP, gripper, and attachment targets for one state."""

    state: AutoState
    tcp_target: PoseTuple
    gripper_target: float
    attachment_mode: AttachmentMode
    failure_message: str | None = None


_NEXT_STATE = {
    AutoState.HOME: AutoState.MOVE_ABOVE_SFP,
    AutoState.MOVE_ABOVE_SFP: AutoState.DESCEND_TO_GRASP,
    AutoState.DESCEND_TO_GRASP: AutoState.CLOSE_GRIPPER,
    AutoState.CLOSE_GRIPPER: AutoState.EXTRACT_FROM_MOUNT,
    AutoState.EXTRACT_FROM_MOUNT: AutoState.LIFT_AFTER_EXTRACTION,
    AutoState.LIFT_AFTER_EXTRACTION: AutoState.TRANSFER_ABOVE_PORT,
    AutoState.TRANSFER_ABOVE_PORT: AutoState.ALIGN_WITH_PORT,
    AutoState.ALIGN_WITH_PORT: AutoState.INSERT_TO_BOTTOM,
    AutoState.INSERT_TO_BOTTOM: AutoState.OPEN_GRIPPER,
    AutoState.OPEN_GRIPPER: AutoState.RETRACT_FROM_PORT,
    AutoState.RETRACT_FROM_PORT: AutoState.LIFT_AFTER_RELEASE,
    AutoState.LIFT_AFTER_RELEASE: AutoState.COMPLETE,
}


def _grasp_pose(frames: ManipulationFrames, sfp_pose: PoseTuple | None = None) -> PoseTuple:
    """Build the SFP grasp pose from the module's reviewed model axes."""
    sfp_pose = frames.sfp_module if sfp_pose is None else sfp_pose
    module_width_axis = _normalize_direction(
        _quat_rotate(sfp_pose.quat_xyzw, (1.0, 0.0, 0.0)),
        field="SFP width axis",
    )
    module_long_axis = _normalize_direction(
        _quat_rotate(sfp_pose.quat_xyzw, (0.0, 1.0, 0.0)),
        field="SFP long axis",
    )
    tool_approach_axis = module_long_axis
    tool_y_axis = (
        tool_approach_axis[1] * module_width_axis[2] - tool_approach_axis[2] * module_width_axis[1],
        tool_approach_axis[2] * module_width_axis[0] - tool_approach_axis[0] * module_width_axis[2],
        tool_approach_axis[0] * module_width_axis[1] - tool_approach_axis[1] * module_width_axis[0],
    )
    pad_center = tuple(
        sfp_pose.xyz[axis] + AUTO_INSERTION.grasp_long_axis_offset_m * module_long_axis[axis] for axis in range(3)
    )
    tcp_position = tuple(
        pad_center[axis] + AUTO_INSERTION.gripper_pad_center_to_tcp_m * tool_approach_axis[axis] for axis in range(3)
    )
    return PoseTuple(
        tcp_position,
        _quaternion_from_axes(module_width_axis, tool_y_axis, tool_approach_axis),
    )


def _offset_pose(pose: PoseTuple, axis: tuple[float, float, float], distance: float) -> PoseTuple:
    """Translate a pose along a reviewed frame direction."""
    return PoseTuple(tuple(pose.xyz[index] + distance * axis[index] for index in range(3)), pose.quat_xyzw)


def _downward_tool_orientation(reference_quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Point tool Z downward while preserving the reference horizontal heading."""
    reference_x = _quat_rotate(reference_quat, (1.0, 0.0, 0.0))
    horizontal_norm = hypot(reference_x[0], reference_x[1])
    if horizontal_norm > 1.0e-8:
        x_axis = (reference_x[0] / horizontal_norm, reference_x[1] / horizontal_norm, 0.0)
    else:
        reference_y = _quat_rotate(reference_quat, (0.0, 1.0, 0.0))
        y_axis = _normalize_direction((reference_y[0], reference_y[1], 0.0), field="tool horizontal heading")
        x_axis = (-y_axis[1], y_axis[0], 0.0)
    z_axis = (0.0, 0.0, -1.0)
    y_axis = (x_axis[1], -x_axis[0], 0.0)
    return _quaternion_from_axes(x_axis, y_axis, z_axis)


class AutomaticInsertionController:
    """Generate deterministic motion commands for the fixed item-0 insertion."""

    def __init__(
        self,
        frames: ManipulationFrames,
        home_pose: PoseTuple,
        *,
        source_grasp_pose: PoseTuple | None = None,
        task_target: TaskTargetConfig = TASK_TARGET,
    ):
        """Initialize state targets from reviewed AIC manipulation frames."""
        if frames.cable_name != task_target.cable_name:
            raise ValueError(f"Automatic insertion expected {task_target.cable_name}, got {frames.cable_name}")

        self._frames = frames
        self._source_grasp_pose = source_grasp_pose
        port_outward_axis = _normalize_direction(
            tuple(frames.port_entrance.xyz[axis] - frames.port_bottom.xyz[axis] for axis in range(3)),
            field="port outward axis",
        )
        transfer_sfp = _offset_pose(
            frames.port_entrance,
            port_outward_axis,
            AUTO_INSERTION.preinsert_port_axis_clearance_m,
        )
        aligned_sfp = _offset_pose(
            frames.port_entrance,
            port_outward_axis,
            AUTO_INSERTION.align_port_axis_clearance_m,
        )
        self._desired_sfp_targets = {
            AutoState.TRANSFER_ABOVE_PORT: transfer_sfp,
            AutoState.ALIGN_WITH_PORT: aligned_sfp,
            AutoState.INSERT_TO_BOTTOM: frames.port_bottom,
            AutoState.OPEN_GRIPPER: frames.port_bottom,
            AutoState.RETRACT_FROM_PORT: frames.port_entrance,
        }
        self._targets = {
            AutoState.HOME: home_pose,
        }
        self._source_sfp_pose = frames.sfp_module
        self._source_rebased = False
        self._set_source_targets(frames.sfp_module)
        self._tool_to_sfp = (
            None
            if source_grasp_pose is None
            else _compose_pose(_inverse_pose(source_grasp_pose), self._source_sfp_pose)
        )
        self._durations = self._safe_state_durations(
            home_pose,
            (
                AutoState.HOME,
                AutoState.MOVE_ABOVE_SFP,
                AutoState.DESCEND_TO_GRASP,
                AutoState.CLOSE_GRIPPER,
            ),
        )
        self.state = AutoState.HOME
        self.attachment_mode = AttachmentMode.MOUNTED
        self._elapsed = 0.0
        self._state_start = home_pose
        self._gripper_start = ROBOT_CONTROL.gripper_open_q
        self._held_tcp_target = home_pose
        self._held_gripper_target = ROBOT_CONTROL.gripper_open_q
        self._failure_message: str | None = None
        self._trajectory_paused = False
        self._paused_translation_error: float | None = None
        self._paused_orientation_error: float | None = None
        self._paused_translation_stall_elapsed = 0.0
        self._paused_orientation_stall_elapsed = 0.0

    @property
    def state_duration(self) -> float:
        """Return the current state's required minimum duration [s]."""
        if self.state in {AutoState.COMPLETE, AutoState.FAILED}:
            return float("inf")
        return self._durations[self.state]

    @property
    def target_pose(self) -> PoseTuple:
        """Return the current state's final TCP target pose [m, XYZW]."""
        return self.target_for(self.state)

    @property
    def tool_to_sfp(self) -> PoseTuple | None:
        """Return the transform captured between the tool and SFP at grasp."""
        return self._tool_to_sfp

    def target_for(self, state: AutoState) -> PoseTuple:
        """Return the frame-derived final TCP target for one state [m, XYZW]."""
        try:
            return self._targets[state]
        except KeyError as error:
            raise ValueError(f"State {state.name} TCP target is unavailable before grasp capture") from error

    def desired_sfp_target_for(self, state: AutoState) -> PoseTuple:
        """Return the desired SFP target for one captured post-grasp state."""
        try:
            return self._desired_sfp_targets[state]
        except KeyError as error:
            raise ValueError(f"State {state.name} does not have a desired SFP target") from error

    def _set_source_targets(
        self,
        sfp_pose: PoseTuple,
        *,
        grasp_reference: PoseTuple | None = None,
    ) -> None:
        """Derive source-side manipulation targets from one SFP module pose."""
        self._source_sfp_pose = sfp_pose
        if self._source_grasp_pose is None:
            grasp_pose = _grasp_pose(self._frames, sfp_pose if grasp_reference is None else grasp_reference)
        else:
            grasp_pose = self._source_grasp_pose
        above_position = tuple(
            grasp_pose.xyz[axis]
            + AUTO_INSERTION.pregrasp_board_normal_clearance_m * self._frames.mount_extraction_axis[axis]
            for axis in range(3)
        )
        above_sfp = PoseTuple(above_position, grasp_pose.quat_xyzw)
        self._targets.update(
            {
                AutoState.MOVE_ABOVE_SFP: above_sfp,
                AutoState.DESCEND_TO_GRASP: grasp_pose,
                AutoState.CLOSE_GRIPPER: grasp_pose,
            }
        )
        extraction_axis = _normalize_direction(
            _quat_rotate(sfp_pose.quat_xyzw, (0.0, 1.0, 0.0)),
            field="SFP extraction axis",
        )
        extracted_sfp = _offset_pose(
            sfp_pose,
            extraction_axis,
            AUTO_INSERTION.sfp_extraction_distance_m,
        )
        self._desired_sfp_targets[AutoState.EXTRACT_FROM_MOUNT] = extracted_sfp
        extracted_tcp_z = grasp_pose.xyz[2] + AUTO_INSERTION.sfp_extraction_distance_m * extraction_axis[2]
        lift_distance = self._targets[AutoState.HOME].xyz[2] - extracted_tcp_z
        if lift_distance <= 0.0:
            raise ValueError("home TCP height must be above the extracted TCP height")
        self._desired_sfp_targets[AutoState.LIFT_AFTER_EXTRACTION] = _offset_pose(
            extracted_sfp,
            (0.0, 0.0, 1.0),
            lift_distance,
        )

    def _rebase_source_targets_once(self, settled_sfp_pose: PoseTuple) -> None:
        """Rebase source-side targets once from the settled SFP module pose."""
        if self._source_rebased:
            return
        self._source_rebased = True
        if self._source_grasp_pose is not None:
            self._source_sfp_pose = settled_sfp_pose
            return
        reviewed_grasp_reference = PoseTuple(
            settled_sfp_pose.xyz,
            self._frames.sfp_module.quat_xyzw,
        )
        self._set_source_targets(settled_sfp_pose, grasp_reference=reviewed_grasp_reference)
        self._durations = self._safe_state_durations(
            self._targets[AutoState.HOME],
            (
                AutoState.HOME,
                AutoState.MOVE_ABOVE_SFP,
                AutoState.DESCEND_TO_GRASP,
                AutoState.CLOSE_GRIPPER,
            ),
        )

    def _safe_state_durations(
        self,
        start_pose: PoseTuple,
        states: tuple[AutoState, ...],
    ) -> dict[AutoState, float]:
        """Return minimum state durations that preserve configured Cartesian limits."""
        durations: dict[AutoState, float] = {}
        previous_target = start_pose
        for state in states:
            profile = AUTO_INSERTION.motion_profile(state.name)
            target = self.target_for(state)
            duration = max(
                profile.duration_s,
                MINIMUM_JERK_PEAK_SPEED_SCALE
                * translation_error(previous_target, target)
                / profile.max_translation_speed_m_s,
                MINIMUM_JERK_PEAK_SPEED_SCALE
                * orientation_error(previous_target, target)
                / profile.max_rotation_speed_rad_s,
            )
            validate_motion_segment(
                distance=translation_error(previous_target, target),
                angle=orientation_error(previous_target, target),
                duration=duration,
                max_translation_speed=profile.max_translation_speed_m_s,
                max_angular_speed=profile.max_rotation_speed_rad_s,
            )
            durations[state] = duration
            previous_target = target
        return durations

    def _capture_tool_to_sfp(self, observation: AutoObservation) -> None:
        """Capture the grasp transform and derive all post-grasp TCP targets."""
        self._tool_to_sfp = _compose_pose(_inverse_pose(observation.tcp_pose), observation.sfp_pose)
        inverse_tool_to_sfp = _inverse_pose(self._tool_to_sfp)
        self._targets.update(
            {
                state: _compose_pose(desired_sfp_pose, inverse_tool_to_sfp)
                for state, desired_sfp_pose in self._desired_sfp_targets.items()
            }
        )
        retract_target = self._targets[AutoState.RETRACT_FROM_PORT]
        self._targets[AutoState.LIFT_AFTER_RELEASE] = PoseTuple(
            (
                retract_target.xyz[0],
                retract_target.xyz[1],
                self._targets[AutoState.HOME].xyz[2],
            ),
            _downward_tool_orientation(retract_target.quat_xyzw),
        )
        self._targets[AutoState.COMPLETE] = self._targets[AutoState.LIFT_AFTER_RELEASE]
        states = tuple(AutoState[name] for name, _ in AUTO_INSERTION.motion_profiles)
        self._durations = self._safe_state_durations(self._targets[AutoState.HOME], states)

    def _gripper_target_for(self, state: AutoState) -> float:
        """Return the final gripper opening target [m]."""
        if state in {
            AutoState.CLOSE_GRIPPER,
            AutoState.EXTRACT_FROM_MOUNT,
            AutoState.LIFT_AFTER_EXTRACTION,
            AutoState.TRANSFER_ABOVE_PORT,
            AutoState.ALIGN_WITH_PORT,
            AutoState.INSERT_TO_BOTTOM,
        }:
            return ROBOT_CONTROL.gripper_closed_q
        return ROBOT_CONTROL.gripper_open_q

    @staticmethod
    def _observation_is_finite(observation: AutoObservation) -> bool:
        """Return whether all observed scalar state is finite."""
        values = (
            *observation.tcp_pose.xyz,
            *observation.tcp_pose.quat_xyzw,
            *observation.sfp_pose.xyz,
            *observation.sfp_pose.quat_xyzw,
            *observation.arm_q,
            *observation.gripper_q,
        )
        return observation.finite and all(isfinite(value) for value in values)

    def _command_for_current_state(self) -> AutoCommand:
        """Build the current state command at its elapsed minimum-jerk fraction."""
        if self.state is AutoState.COMPLETE:
            self._held_tcp_target = self.target_pose
            self._held_gripper_target = ROBOT_CONTROL.gripper_open_q
            return AutoCommand(
                self.state,
                self._held_tcp_target,
                self._held_gripper_target,
                self.attachment_mode,
            )
        duration = self.state_duration
        alpha = min(1.0, self._elapsed / duration)
        tcp_target = interpolate_pose(self._state_start, self.target_pose, alpha)
        final_gripper_target = self._gripper_target_for(self.state)
        gripper_target = self._gripper_start + minimum_jerk(alpha) * (final_gripper_target - self._gripper_start)
        self._held_tcp_target = tcp_target
        self._held_gripper_target = gripper_target
        return AutoCommand(self.state, tcp_target, gripper_target, self.attachment_mode)

    def _fail(self, message: str) -> AutoCommand:
        """Enter the terminal failed state while preserving the current targets."""
        self.state = AutoState.FAILED
        self.attachment_mode = AttachmentMode.FAILED
        self._failure_message = message
        return AutoCommand(
            self.state,
            self._held_tcp_target,
            self._held_gripper_target,
            self.attachment_mode,
            self._failure_message,
        )

    def _tracking_errors(self, observation: AutoObservation) -> tuple[float, float]:
        """Return observed errors against the last issued TCP target."""
        return (
            translation_error(observation.tcp_pose, self._held_tcp_target),
            orientation_error(observation.tcp_pose, self._held_tcp_target),
        )

    def _reset_pause_tracking(self) -> None:
        """Clear paused trajectory progress tracking."""
        self._trajectory_paused = False
        self._paused_translation_error = None
        self._paused_orientation_error = None
        self._paused_translation_stall_elapsed = 0.0
        self._paused_orientation_stall_elapsed = 0.0

    @staticmethod
    def _paused_axis_stall_elapsed(
        *,
        error: float,
        previous_error: float | None,
        elapsed: float,
        resume_error: float,
        min_progress_speed: float,
        dt: float,
    ) -> float:
        """Return a per-axis paused no-progress timer update."""
        if error <= resume_error:
            return 0.0
        if previous_error is None:
            return 0.0
        if previous_error - error >= min_progress_speed * dt:
            return 0.0
        return elapsed + dt

    def _pause_or_fail(self, observation: AutoObservation, *, dt: float) -> AutoCommand | None:
        """Pause elapsed advancement when observed tracking is outside the catch-up envelope."""
        translation, orientation = self._tracking_errors(observation)
        if self._trajectory_paused:
            if (
                translation <= AUTO_INSERTION.trajectory_translation_resume_error_m
                and orientation <= AUTO_INSERTION.trajectory_orientation_resume_error_rad
            ):
                self._reset_pause_tracking()
                return None
        elif (
            translation <= AUTO_INSERTION.trajectory_translation_pause_error_m
            and orientation < AUTO_INSERTION.trajectory_orientation_pause_error_rad
        ):
            self._paused_translation_error = translation
            self._paused_orientation_error = orientation
            self._paused_translation_stall_elapsed = 0.0
            self._paused_orientation_stall_elapsed = 0.0
            return None
        else:
            self._trajectory_paused = True
            self._paused_translation_error = translation
            self._paused_orientation_error = orientation
            self._paused_translation_stall_elapsed = 0.0
            self._paused_orientation_stall_elapsed = 0.0
            return AutoCommand(
                self.state,
                self._held_tcp_target,
                self._held_gripper_target,
                self.attachment_mode,
            )

        self._paused_translation_stall_elapsed = self._paused_axis_stall_elapsed(
            error=translation,
            previous_error=self._paused_translation_error,
            elapsed=self._paused_translation_stall_elapsed,
            resume_error=AUTO_INSERTION.trajectory_translation_resume_error_m,
            min_progress_speed=AUTO_INSERTION.trajectory_translation_min_progress_m_s,
            dt=dt,
        )
        self._paused_orientation_stall_elapsed = self._paused_axis_stall_elapsed(
            error=orientation,
            previous_error=self._paused_orientation_error,
            elapsed=self._paused_orientation_stall_elapsed,
            resume_error=AUTO_INSERTION.trajectory_orientation_resume_error_rad,
            min_progress_speed=AUTO_INSERTION.trajectory_orientation_min_progress_rad_s,
            dt=dt,
        )
        self._paused_translation_error = translation
        self._paused_orientation_error = orientation
        stalled_axes = []
        if (
            translation > AUTO_INSERTION.trajectory_translation_resume_error_m
            and self._paused_translation_stall_elapsed >= AUTO_INSERTION.trajectory_stall_timeout_s
        ):
            stalled_axes.append("translation")
        if (
            orientation > AUTO_INSERTION.trajectory_orientation_resume_error_rad
            and self._paused_orientation_stall_elapsed >= AUTO_INSERTION.trajectory_stall_timeout_s
        ):
            stalled_axes.append("orientation")
        if stalled_axes:
            return self._fail(
                f"{self.state.name}: {' and '.join(stalled_axes)} stalled "
                f"with translation error {translation:.6f} m "
                f"and orientation error {orientation:.6f} rad"
            )
        return AutoCommand(
            self.state,
            self._held_tcp_target,
            self._held_gripper_target,
            self.attachment_mode,
        )

    @staticmethod
    def _within_tolerance(
        actual: PoseTuple,
        target: PoseTuple,
        *,
        translation_tolerance: float,
        orientation_tolerance: float,
    ) -> bool:
        """Return whether a pose meets translation [m] and orientation [rad] bounds."""
        return (
            translation_error(actual, target) <= translation_tolerance
            and orientation_error(actual, target) <= orientation_tolerance
        )

    def _can_transition(self, observation: AutoObservation) -> bool:
        """Return whether the current state reached all required observed targets."""
        if not self._within_tolerance(
            observation.tcp_pose,
            self.target_pose,
            translation_tolerance=AUTO_INSERTION.tcp_translation_tolerance_m,
            orientation_tolerance=AUTO_INSERTION.tcp_orientation_tolerance_rad,
        ):
            return False
        if self.state is AutoState.CLOSE_GRIPPER:
            return (
                len(observation.gripper_q) == 2
                and all(
                    abs(position - ROBOT_CONTROL.gripper_closed_q) <= AUTO_INSERTION.gripper_position_tolerance_m
                    for position in observation.gripper_q
                )
                and self._within_tolerance(
                    observation.sfp_pose,
                    self._source_sfp_pose,
                    translation_tolerance=AUTO_INSERTION.grasp_translation_tolerance_m,
                    orientation_tolerance=AUTO_INSERTION.grasp_orientation_tolerance_rad,
                )
            )
        if self.state is AutoState.INSERT_TO_BOTTOM:
            return True
        if self.state is AutoState.OPEN_GRIPPER:
            seated = self._within_tolerance(
                observation.sfp_pose,
                self._desired_sfp_targets[AutoState.INSERT_TO_BOTTOM],
                translation_tolerance=AUTO_INSERTION.seat_translation_tolerance_m,
                orientation_tolerance=AUTO_INSERTION.seat_orientation_tolerance_rad,
            )
            gripper_open = len(observation.gripper_q) == 2 and all(
                abs(position - ROBOT_CONTROL.gripper_open_q) <= AUTO_INSERTION.gripper_position_tolerance_m
                for position in observation.gripper_q
            )
            return seated and gripper_open
        return True

    def _advance_state(self) -> None:
        """Enter the next state and preserve final targets as its trajectory start."""
        previous_state = self.state
        self._state_start = self.target_pose
        self._gripper_start = self._gripper_target_for(previous_state)
        if previous_state is AutoState.CLOSE_GRIPPER:
            self.attachment_mode = AttachmentMode.GRASPED
        elif previous_state is AutoState.INSERT_TO_BOTTOM:
            self.attachment_mode = AttachmentMode.SEATED
        self.state = _NEXT_STATE[previous_state]
        self._elapsed = 0.0
        self._reset_pause_tracking()

    def command(self, observation: AutoObservation, *, dt: float) -> AutoCommand:
        """Advance deterministic motion from one observed simulation state.

        Args:
            observation: Current ground-truth TCP, SFP, arm, and gripper state.
            dt: Controller step duration [s].
        """
        if self.state is AutoState.FAILED:
            return AutoCommand(
                self.state,
                self._held_tcp_target,
                self._held_gripper_target,
                self.attachment_mode,
                self._failure_message,
            )
        if self.state is AutoState.COMPLETE:
            return AutoCommand(
                self.state,
                self._held_tcp_target,
                self._held_gripper_target,
                self.attachment_mode,
            )
        if not isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not self._observation_is_finite(observation):
            return self._fail(f"{self.state.name}: non-finite observation")

        paused_command = self._pause_or_fail(observation, dt=dt)
        if paused_command is not None:
            return paused_command

        self._elapsed += dt
        command = self._command_for_current_state()
        translation = translation_error(observation.tcp_pose, self.target_pose)
        orientation = orientation_error(observation.tcp_pose, self.target_pose)
        if self._elapsed >= self.state_duration and self._can_transition(observation):
            if self.state is AutoState.HOME:
                self._rebase_source_targets_once(observation.sfp_pose)
            elif self.state is AutoState.CLOSE_GRIPPER:
                self._capture_tool_to_sfp(observation)
            self._advance_state()
            return self._command_for_current_state()
        if self._elapsed >= self.state_duration + AUTO_INSERTION.state_timeout_margin:
            return self._fail(
                f"{self.state.name}: timeout with translation error {translation:.6f} m "
                f"and orientation error {orientation:.6f} rad"
            )
        return command
