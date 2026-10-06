"""
Fixed test harness for the vortex outflow experiment. DO NOT MODIFY.

A Lamb-Oseen vortex is convected by a uniform flow through the right boundary
of a 2D domain (Wissocq et al., J. Comput. Phys. 2017). A perfect outflow
boundary lets the vortex and all acoustic waves leave without reflection.

The candidate outflow boundary is compared against a reference solution on a
periodic domain that is long enough for nothing to come back into the
measurement window during the run, i.e. the "infinite domain" answer.

    uv run --extra cpu python autoresearch/vortex_cbc/prepare.py

computes and caches the reference once. evaluate.py then scores a candidate.
"""
import math
import time
from pathlib import Path

import numpy as np
import torch

import lettuce as lt

# --------------------------------------------------------------------------
# Fixed experiment constants
# --------------------------------------------------------------------------
NX = 100                 # test domain length (x), outlet at x = NX - 1
NY = 100                 # domain height (y), periodic in y
REYNOLDS = 500.0         # based on mean flow velocity and NY
MACH = 0.1               # mean flow Mach number
VORTEX_BETA = 0.5        # vortex strength relative to the mean flow
VORTEX_RADIUS = 10.0     # vortex core radius in lattice units
NUM_STEPS = 2000         # vortex fully leaves the domain after ~1400 steps
SAMPLE_EVERY = 20        # compare with the reference every N steps
TIME_LIMIT = 300.0       # seconds; slower candidates are rejected

CS = 1.0 / math.sqrt(3.0)
# Reference domain: acoustic waves travel CS * NUM_STEPS cells during the run;
# they must not wrap around the periodic domain into the measurement window.
NX_REF = NX + 2 * int(math.ceil(CS * NUM_STEPS)) + 64
REF_OFFSET = (NX_REF - NX) // 2   # x index of test column 0 in the reference

CACHE_DIR = Path(__file__).parent / ".cache"
REFERENCE_FILE = CACHE_DIR / f"reference_{NX}x{NY}_{NUM_STEPS}.npz"


def make_context() -> lt.Context:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return lt.Context(device=device, dtype=torch.float64, use_native=False)


class ConvectedVortex(lt.ExtFlow):
    """Isothermal Lamb-Oseen vortex in a uniform flow along +x.

    Units are fixed to NY, so the test and the (longer) reference domain share
    the same relaxation time. Boundaries are created by `boundary_factory`
    (flow -> list of boundaries); None gives a fully periodic domain.
    """

    def __init__(self, context, nx, xc, boundary_factory=None):
        self.xc = xc
        self._boundary_factory = boundary_factory
        self._boundaries = None
        self.initialize_fneq = True
        super().__init__(context, [nx, NY], REYNOLDS, MACH, lt.D2Q9())

    def make_resolution(self, resolution, stencil=None):
        return list(resolution)

    def make_units(self, reynolds_number, mach_number, resolution):
        return lt.UnitConversion(
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=NY, characteristic_length_pu=1,
            characteristic_velocity_pu=1)

    def initial_pu(self):
        nx, ny = self.resolution
        x, y = torch.meshgrid(
            torch.arange(nx, device=self.context.device, dtype=self.context.dtype),
            torch.arange(ny, device=self.context.device, dtype=self.context.dtype),
            indexing="ij")
        u0 = self.units.convert_velocity_to_lu(1.0)
        dx, dy = x - self.xc, y - 0.5 * ny
        g = torch.exp(-(dx ** 2 + dy ** 2) / (2 * VORTEX_RADIUS ** 2))
        ux = u0 - VORTEX_BETA * u0 * dy / VORTEX_RADIUS * g
        uy = VORTEX_BETA * u0 * dx / VORTEX_RADIUS * g
        # isothermal radial equilibrium: cs^2 d(ln rho)/dr = u_theta^2 / r
        rho = torch.exp(-(VORTEX_BETA * u0) ** 2 / (2 * CS ** 2) * g ** 2)
        p = self.units.convert_density_lu_to_pressure_pu(rho)
        u = self.units.convert_velocity_to_pu(torch.stack([ux, uy]))
        return p, u

    @property
    def post_boundaries(self):
        if self._boundaries is None:
            self._boundaries = ([] if self._boundary_factory is None
                                else list(self._boundary_factory(self)))
        return self._boundaries


def inlet(flow: ConvectedVortex) -> lt.Boundary:
    """Fixed inflow on the left: uniform velocity, reference pressure."""
    mask = torch.zeros(flow.resolution, dtype=torch.bool, device=flow.context.device)
    mask[0, :] = True
    return lt.EquilibriumBoundaryPU(flow.context, flow, mask, velocity=[1.0, 0.0], pressure=0.0)


def _snapshot(flow, x_slice):
    rho = flow.rho()[0, x_slice, :]
    u = flow.u()[:, x_slice, :]
    return (flow.context.convert_to_ndarray(rho),
            flow.context.convert_to_ndarray(u))


def _run(flow, x_slice, deadline=None):
    """Run NUM_STEPS and sample rho, u in x_slice. Returns (rho[t], u[t]) or None on blow-up/timeout."""
    sim = lt.Simulation(flow, lt.BGKCollision(tau=flow.units.relaxation_parameter_lu), reporter=[])
    rhos, us = [], []
    for _ in range(NUM_STEPS // SAMPLE_EVERY):
        sim(SAMPLE_EVERY)
        rho, u = _snapshot(flow, x_slice)
        if not (np.isfinite(rho).all() and np.isfinite(u).all()):
            return None, "diverged"
        if deadline is not None and time.perf_counter() > deadline:
            return None, "timeout"
        rhos.append(rho)
        us.append(u)
    return (np.stack(rhos), np.stack(us)), "ok"


def reference():
    """Load the cached reference solution, computing it on first use."""
    if not REFERENCE_FILE.exists():
        CACHE_DIR.mkdir(exist_ok=True)
        print(f"computing reference on {NX_REF}x{NY} (once) ...", flush=True)
        t0 = time.perf_counter()
        flow = ConvectedVortex(make_context(), NX_REF, REF_OFFSET + NX // 2)
        (rho, u), _ = _run(flow, slice(REF_OFFSET, REF_OFFSET + NX))
        np.savez_compressed(REFERENCE_FILE, rho=rho, u=u)
        print(f"reference done in {time.perf_counter() - t0:.0f} s -> {REFERENCE_FILE}")
    data = np.load(REFERENCE_FILE)
    return data["rho"], data["u"]


def evaluate(make_outlet) -> dict:
    """Score an outflow boundary. `make_outlet(flow)` returns a lettuce Boundary
    applied as post-collision boundary on the right edge (x = NX - 1).

    score = (err_rho + err_u) / 2, lower is better, where both errors are the
    RMS deviation from the reference over all samples and interior nodes,
    normalised by the vortex amplitude (density dip / peak swirl velocity).
    """
    rho_ref, u_ref = reference()
    flow = ConvectedVortex(make_context(), NX, NX // 2,
                           boundary_factory=lambda f: [inlet(f), make_outlet(f)])
    t0 = time.perf_counter()
    result, status = _run(flow, slice(0, NX), deadline=t0 + TIME_LIMIT)
    seconds = time.perf_counter() - t0
    if result is None:
        return {"status": status, "score": float("inf"), "seconds": seconds}

    rho, u = result
    interior = slice(1, NX - 1)   # boundary columns are excluded
    u0 = flow.units.convert_velocity_to_lu(1.0)
    rho_scale = 1.0 - math.exp(-(VORTEX_BETA * u0) ** 2 / (2 * CS ** 2))
    u_scale = VORTEX_BETA * u0 * math.exp(-0.5)
    d_rho = rho[:, interior] - rho_ref[:, interior]
    d_u = u[:, :, interior] - u_ref[:, :, interior]
    err_rho = float(np.sqrt(np.mean(d_rho ** 2)) / rho_scale)
    err_u = float(np.sqrt(np.mean(np.sum(d_u ** 2, axis=1))) / u_scale)
    # error after the vortex has left: what remains is pure reflection
    late = slice(len(rho) * 3 // 4, None)
    err_rho_late = float(np.sqrt(np.mean(d_rho[late] ** 2)) / rho_scale)
    return {"status": "ok", "score": 0.5 * (err_rho + err_u),
            "err_rho": err_rho, "err_u": err_u, "err_rho_late": err_rho_late,
            "seconds": seconds}


if __name__ == "__main__":
    rho, _ = reference()
    print(f"reference ready: {rho.shape[0]} samples of {rho.shape[1]}x{rho.shape[2]}")
