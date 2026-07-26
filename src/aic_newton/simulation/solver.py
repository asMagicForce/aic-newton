"""Build the coupled rigid-body and cable solver."""

import newton
import newton.examples
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

from ..config import SOLVER


def _build_coupled_solver(
    *,
    model: newton.Model,
    robot_bodies: list[int],
    robot_joints: list[int],
    proxy_bodies: list[int],
    payload_bodies: list[int],
    payload_joints: list[int],
    vbd_iterations: int,
) -> SolverCoupledProxy:
    """Build the validated lagged MuJoCo-to-VBD proxy coupling."""
    return SolverCoupledProxy(
        model=model,
        entries=[
            SolverCoupled.Entry(
                name="mjc",
                solver=lambda view: SolverMuJoCo(
                    model=view,
                    solver="newton",
                    integrator="implicitfast",
                    cone="elliptic",
                    iterations=SOLVER.mujoco_iterations,
                    ls_iterations=SOLVER.mujoco_line_search_iterations,
                    use_mujoco_contacts=False,
                    njmax=SOLVER.mujoco_njmax,
                    nconmax=SOLVER.mujoco_nconmax,
                ),
                bodies=robot_bodies,
                joints=robot_joints,
            ),
            SolverCoupled.Entry(
                name="vbd",
                solver=lambda view: SolverVBD(
                    model=view,
                    iterations=vbd_iterations,
                    rigid_contact_history=False,
                    rigid_body_contact_buffer_size=SOLVER.rigid_body_contact_buffer_size,
                ),
                bodies=payload_bodies,
                joints=payload_joints,
            ),
        ],
        coupling=SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(
                    source="mjc",
                    destination="vbd",
                    bodies=proxy_bodies,
                    mass_scale=SOLVER.proxy_mass_scale,
                    mode="lagged",
                    collision_pipeline=lambda view: newton.examples.create_collision_pipeline(
                        view,
                        broad_phase="explicit",
                    ),
                    collide_interval=SOLVER.proxy_collide_interval,
                )
            ],
            iterations=SOLVER.coupling_iterations,
        ),
    )
