"""Define command-line options for the AIC simulation."""

import argparse

import newton.examples

from .config import SIMULATION, TASK_TARGET, VIEWER
from .simulation.cameras import nonnegative_finite_float


def nonnegative_int(value: str) -> int:
    """Parse a nonnegative integer command-line value."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def create_parser():
    """Create the standalone simulation argument parser."""
    parser = newton.examples.create_parser()
    parser.add_argument("--auto", action="store_true", help="Run the automatic SFP insertion sequence.")
    parser.add_argument("--camera", action="store_true", help="Show the three Basler sensor images.")
    parser.add_argument(
        "--cable-index",
        type=nonnegative_int,
        default=TASK_TARGET.cable_index,
        help="Zero-based cable assembly index.",
    )
    parser.add_argument(
        "--nic-card-index",
        type=nonnegative_int,
        default=TASK_TARGET.nic_card_index,
        help="Zero-based NIC card index.",
    )
    parser.add_argument(
        "--nic-port-index",
        type=nonnegative_int,
        default=TASK_TARGET.nic_port_index,
        help="Zero-based SFP port index on the selected NIC card.",
    )
    parser.add_argument(
        "--camera-speed",
        type=nonnegative_finite_float,
        default=VIEWER.camera_speed_m_s,
        help="Viewer WASD/QE translation speed in m/s.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=SIMULATION.default_substeps,
        help="Simulation substeps per rendered frame.",
    )
    parser.add_argument(
        "--no-graph-capture",
        action="store_false",
        dest="graph_capture",
        default=True,
        help="Disable graph capture.",
    )
    return parser
