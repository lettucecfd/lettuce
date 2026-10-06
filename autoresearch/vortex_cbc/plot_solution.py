"""
Plot the solution of the current boundary.py next to the reference: density
deviation of the reference and the candidate, their difference, and the error
over time compared with the baseline outlet. Writes solution.png.

    uv run --extra cpu python plot_solution.py [solution.png]
"""
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import lettuce as lt
import prepare as P
from boundary import make_outlet

HERE = Path(__file__).parent
CURRENT, BASELINE, TEXT, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"


def run(outlet):
    flow = P.ConvectedVortex(P.make_context(), P.NX, P.NX // 2,
                             boundary_factory=lambda f: [P.inlet(f), outlet(f)])
    (rho, u), status = P._run(flow, slice(0, P.NX))
    return rho, flow


rho_ref, _ = P.reference()
rho, flow = run(make_outlet)
rho_base, _ = run(lambda f: lt.EquilibriumOutletP(direction=[1, 0], flow=f, rho_outlet=1.0))

u0 = flow.units.convert_velocity_to_lu(1.0)
rho_scale = 1.0 - math.exp(-(P.VORTEX_BETA * u0) ** 2 / (2 * P.CS ** 2))
steps = (np.arange(len(rho_ref)) + 1) * P.SAMPLE_EVERY
inner = slice(1, P.NX - 1)


def err(r):
    return np.sqrt(((r[:, inner] - rho_ref[:, inner]) ** 2).mean(axis=(1, 2))) / rho_scale


samples = [24, 44, 54, 74]                     # steps 500, 900, 1100, 1500
# the boundary columns are not part of the metric; the inlet column also holds
# populations streamed in periodically from the outlet. Hide both (gray).
def interior(a):
    a = a.astype(float).copy()
    a[:, 0] = a[:, -1] = np.nan
    return a
diff = interior(rho - rho_ref)
lim_field = np.abs(rho_ref - 1).max()
lim_diff = np.nanmax(np.abs(diff[samples]))

fig = plt.figure(figsize=(13, 12.5), constrained_layout=True)
grid = fig.add_gridspec(4, len(samples), height_ratios=[1, 1, 1, 0.9])
rows = [("reference  ρ − 1", rho_ref - 1, lim_field, "RdBu_r"),
        ("boundary.py  ρ − 1", interior(rho - 1), lim_field, "RdBu_r"),
        ("difference  ρ − ρ_ref", diff, lim_diff, "PuOr_r")]
for k, (label, data, lim, cmap) in enumerate(rows):
    for j, t in enumerate(samples):
        ax = fig.add_subplot(grid[k, j])
        ax.set_facecolor("#d9d8d3")
        img = ax.imshow(data[t].T, origin="lower", cmap=cmap, vmin=-lim, vmax=lim)
        ax.axvline(P.NX - 1, color="0.3", lw=1, ls="--")
        ax.set_xticks([0, P.NX // 2, P.NX - 1])
        ax.set_yticks([0, P.NY // 2, P.NY - 1])
        if k == 0:
            ax.set_title(f"step {steps[t]}", color=TEXT)
        if j == 0:
            ax.set_ylabel(f"{label}\n\ny", color=TEXT)
    fig.colorbar(img, ax=fig.axes[-len(samples):], shrink=0.85,
                 label=f"{label.split('  ')[1]}  (±{lim:.1e})")

ax = fig.add_subplot(grid[3, :])
e_cur, e_base = err(rho), err(rho_base)
ax.plot(steps, e_base, color=BASELINE, lw=2)
ax.plot(steps, e_cur, color=CURRENT, lw=2)
ax.text(steps[np.argmax(e_base)] + 30, e_base.max(), "baseline EquilibriumOutletP",
        color=TEXT, va="center", fontsize=9)
ax.text(steps[np.argmax(e_cur)] + 30, e_cur.max(), "boundary.py", color=TEXT,
        va="bottom", fontsize=9)
for t in samples:
    ax.axvline(steps[t], color=GRID, lw=1, zorder=0)
ax.set_yscale("log")
ax.set_xlabel("time step", color=TEXT)
ax.set_ylabel("RMS density error\n/ vortex density dip", color=TEXT)
ax.tick_params(colors=MUTED)
ax.grid(True, which="major", color=GRID, lw=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.suptitle(f"Current boundary vs. reference (boundary columns hidden) — RMS density error: "
             f"boundary.py {np.sqrt((e_cur ** 2).mean()):.2e}, "
             f"baseline {np.sqrt((e_base ** 2).mean()):.2e}", fontsize=13, color=TEXT)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "solution.png"
fig.savefig(out, dpi=100)
print(f"written {out}")
