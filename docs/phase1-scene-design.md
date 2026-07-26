# AIC Phase 1 Scene Design

## Goal

Provide a deterministic Phase 1 task-board scene with five complete cable
assemblies and the robot at home. The default interactive mode starts with an
empty gripper. Automatic mode keeps both ends of cable 0 in their Phase 1
mounts while the gripper approaches the SFP, then transfers only the SFP end
to the gripper.

This milestone adds the deterministic cable-0 approach, gripper closure,
attachment handoff, mount extraction, and post-extraction lift. Port
insertion, insertion scoring, cable selection, and layout randomization remain
out of scope.

## Authoritative Sources

- `intrinsic-dev/aic`, branch `phase_1`, defines the official task-board
  geometry, component frame formulas, and `sfp_sc_cable_phase_1` model.
- `intrinsic-dev/aic-phase-1` defines the five-cable task and scene invariants.
- `/home/sustechdl/Documents/aic_pose` provides a reviewed deterministic
  instance of the legal Phase 1 rail layout.

The Newton repository vendors every required runtime asset. It must not depend
on any of these source directories after asset preparation.

## Fixed Layout

Use the reviewed fixed task-board pose and paired mount translations derived
from the `sample_004` Phase 1 pose scene:

- Task board: `(x, y, z) = (0.25617, 0.047549, 1.14)` m.
- Task-board orientation: `(roll, pitch, yaw) = (0, 0, 2.77239)` rad.
- Cable pair translations:
  `(-0.080948, -0.012195, 0.07793, -0.009757, 0.083624)` m.
- Cable pair rails: `(0, 0, 0, 1, 1)`.
- Pair rotations are zero.

For each index, the SFP mount and SC mount use the same rail translation.
The SFP mount occupies the far mounting slot at task-board-local
`x = 0.01` m, leaving the middle slot clear between it and the SC mount.
Because the reviewed cable curve was recorded with the SFP mount at
`x = 0.055` m, its centerline is smoothly retargeted between the relocated
SFP end and the unchanged, reviewed SC-plug pose. Both connector end
segments remain normal to their mounts. An `80 mm` cubic transition blends
each prescribed endpoint tangent back into the reviewed centerline without
introducing a sharp rest bend. The cable remains sampled at no more than
`10 mm` per segment.
NIC-card rail translations are `(0, 0, 0, 0, 0)` m because the five cards
occupy distinct parallel rails. SC-port translations are
`(-0.07, 0, 0.07, -0.04, 0.04)` m: the first three ports occupy the first
official SC row and the last two occupy the second row.

## Scene Contents

The task board contains exactly:

- Five cable assemblies named `cable_0` through `cable_4`.
- Five SFP mounts named `sfp_mount_0` through `sfp_mount_4`.
- Five SC mounts named `sc_mount_0` through `sc_mount_4`.
- Five SFP modules and five SC plugs, one pair per cable.
- Five SC ports named `sc_port_0` through `sc_port_4`.

Automatic mode derives the grasp TCP directly from the mounted cable-0 LC
pose and the reviewed qualification-phase tool-to-LC transform. Both the SFP
and SC connector bodies start on fixed joints to kinematic mount anchors, so
cable tension cannot pull them through the fixtures. The gripper first
reaches an offset pose with the final grasp orientation, then descends by
translation only. It then closes and atomically transfers the SFP endpoint
from the mount fixed joint to the gripper fixed joint. The SC endpoint keeps
the same fixed-joint attachment to its mount. The SFP then withdraws `33 mm`
along its local `+Y` long axis before the grasped assembly lifts along world
`+Z` until the TCP reaches its home height. The SFP then transfers above NIC
card 0, aligns with SFP port 0 at the port entrance, and inserts `45.8 mm`
along the port axis to its geometric seat. At the seat, ownership transfers
atomically from the gripper fixed joint to the NIC fixed joint before the
fingers open. The released gripper retreats to the entrance and lifts along
world `+Z` until the TCP reaches its home height. No attachment mode directly
projects or overwrites the connector body transform.
The port target follows the AIC frame convention: rotate the raw SDF port frame
by `Rx(-90 deg)` to obtain the SFP module orientation, then offset the module
origin so its local `-Y` mating face at `23.65 mm` lands on the port entrance.
The fixed-joint grasp filters the tool and finger collision geometry from the
SFP and SC connector bodies. It also filters the tool from the first
`144 mm` of cable represented by the rigid connector strain-relief region;
the remaining flexible cable and all environment contacts stay enabled.
- Five NIC card mounts and five NIC cards, indexed from 0 through 4.

The enclosure, floor, robot, wrist cameras, force-torque sensor, and gripper
remain unchanged.

## Cable Initialization

Each cable uses the official Phase 1 cable-to-SFP-mount transform and the
official plug transforms. Its curve is resampled at a maximum segment length
of 10 mm, preserving the existing Newton cable geometry convention.

All five cable assemblies start static, matching the Phase 1
`CableModeratorPlugin` behavior. Neither cable end is attached to the robot.
Static initialization prevents the inactive cables from falling out of their
mounts and avoids paying for five fully dynamic rods before a cable is chosen.

The scene representation groups each cable's plugs, rod segments, mount pair,
and identifying labels so a later feature can replace one static assembly with
a VBD assembly without changing scene naming or placement.

## Solver Boundary

The MuJoCo solver continues to control the UR5e and gripper. The initial Phase 1
scene adds no cable body to the MuJoCo proxy coupling and creates no
tool-to-plug attachment. Static task-board components and cable assemblies
participate in rendering and may provide collision geometry where required,
but they do not add dynamic degrees of freedom in this change.

## Assets

Vendor the Phase 1 versions of the SFP mount, SC mount, SC ports, NIC card
mounts, cable plugs, and cable model assets under `assets/aic_assets`. Preserve
their upstream license and source attribution. Do not restore the removed
task-board CAD source files; runtime GLB and model metadata are sufficient.

## Validation

Automated tests verify:

- Exact counts and indexed labels for all Phase 1 components.
- The fixed board pose and five reviewed pair translations.
- SFP and SC mounts in each pair share a rail translation.
- Cable endpoints match their official mount-relative poses.
- No cable or plug is attached to the gripper at initialization.
- No two NIC cards or SC ports overlap in the fixed layout.
- Model state remains finite during a short simulation.

Manual Viewer validation verifies that all five cable assemblies, mounts, NIC
cards, and SC ports are visible and correctly aligned, and that the robot
starts at home with an empty gripper.
