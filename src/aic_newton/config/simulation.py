"""Configure simulation timing, contacts, and solver settings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Configure frame timing and global model properties."""

    fps: int = 60
    default_substeps: int = 8
    gravity_m_s2: float = -9.81
    rigid_contact_max: int = 16384
    rigid_gap_m: float = 0.001

    def __post_init__(self) -> None:
        """Validate timing and contact capacities."""
        if self.fps <= 0 or self.default_substeps <= 0 or self.rigid_contact_max <= 0:
            raise ValueError("Simulation rates and contact capacity must be positive")


@dataclass(frozen=True)
class SolverConfig:
    """Configure MuJoCo, VBD, and lagged proxy coupling."""

    vbd_iterations: int = 8
    rigid_body_contact_buffer_size: int = 1024
    mujoco_iterations: int = 100
    mujoco_line_search_iterations: int = 20
    mujoco_njmax: int = 512
    mujoco_nconmax: int = 1024
    coupling_iterations: int = 1
    proxy_mass_scale: float = 1.0
    proxy_collide_interval: int = 1

    def __post_init__(self) -> None:
        """Validate solver iteration and capacity settings."""
        values = (
            self.vbd_iterations,
            self.rigid_body_contact_buffer_size,
            self.mujoco_iterations,
            self.mujoco_line_search_iterations,
            self.mujoco_njmax,
            self.mujoco_nconmax,
            self.coupling_iterations,
            self.proxy_collide_interval,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Solver iterations and capacities must be positive")


SIMULATION = SimulationConfig()
SOLVER = SolverConfig()
