"""
Plot the solution of the current boundary.py next to the reference, once for
the density deviation and once for the vorticity: reference, candidate, their
difference, and the error over time compared with the baseline outlet.
Writes solution.png (density) and solution_vorticity.png.

    uv run --extra cpu python plot_solution.py [solution.png]
"""
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
SAMPLES = [24, 44, 54, 74]                      # steps 500, 900, 1100, 1500
STEPS = (np.arange(P.NUM_STEPS // P.SAMPLE_EVERY) + 1) * P.SAMPLE_EVERY


def run(outlet):
    flow = P.ConvectedVortex(P.make_context(), P.NX, P.NX // 2,
                             boundary_factory=lambda f: [P.inlet(f), outlet(f)])
    (rho, u), _ = P._run(flow, slice(0, P.NX))
    return rho, u


def vorticity(u):
    return np.gradient(u[:, 1], axis=1) - np.gradient(u[:, 0], axis=2)


def hide(a, columns):
    """Gray out columns that are not part of the comparison."""
    a = a.astype(float).copy()
    a[:, columns] = np.nan
    return a


def plot(ref, cur, base, hidden, name, symbol, diff_symbol, out):
    """ref, cur, base: field over time, shape (samples, NX, NY)."""
    cur, base = hide(cur, hidden), hide(base, hidden)
    diff = cur - ref
    scale = np.abs(ref[0]).max()
    lim_field = np.abs(ref).max()
    lim_diff = np.nanmax(np.abs(diff[SAMPLES]))

    fig = plt.figure(figsize=(13, 12.5), constrained_layout=True)
    grid = fig.add_gridspec(4, len(SAMPLES), height_ratios=[1, 1, 1, 0.9])
    rows = [(f"reference  {symbol}", ref, lim_field, "RdBu_r"),
            (f"boundary.py  {symbol}", cur, lim_field, "RdBu_r"),
            (f"difference  {diff_symbol}", diff, lim_diff, "PuOr_r")]
    for k, (label, data, lim, cmap) in enumerate(rows):
        for j, t in enumerate(SAMPLES):
            ax = fig.add_subplot(grid[k, j])
            ax.set_facecolor("#d9d8d3")
            img = ax.imshow(data[t].T, origin="lower", cmap=cmap, vmin=-lim, vmax=lim)
            ax.axvline(P.NX - 1, color="0.3", lw=1, ls="--")
            ax.set_xticks([0, P.NX // 2, P.NX - 1])
            ax.set_yticks([0, P.NY // 2, P.NY - 1])
            if k == 0:
                ax.set_title(f"step {STEPS[t]}", color=TEXT)
            if j == 0:
                ax.set_ylabel(f"{label}\n\ny", color=TEXT)
        fig.colorbar(img, ax=fig.axes[-len(SAMPLES):], shrink=0.85,
                     label=f"{label.split('  ')[1]}  (±{lim:.1e})")

    def err(field):
        return np.sqrt(np.nanmean((field - ref) ** 2, axis=(1, 2))) / scale

    e_cur, e_base = err(cur), err(base)
    ax = fig.add_subplot(grid[3, :])
    ax.plot(STEPS, e_base, color=BASELINE, lw=2)
    ax.plot(STEPS, e_cur, color=CURRENT, lw=2)
    ax.text(STEPS[np.argmax(e_base)] + 30, e_base.max(), "baseline EquilibriumOutletP",
            color=TEXT, va="center", fontsize=9)
    ax.text(STEPS[np.argmax(e_cur)] + 30, e_cur.max(), "boundary.py", color=TEXT,
            va="bottom", fontsize=9)
    for t in SAMPLES:
        ax.axvline(STEPS[t], color=GRID, lw=1, zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("time step", color=TEXT)
    ax.set_ylabel(f"RMS {name} error\n/ initial peak |{symbol}|", color=TEXT)
    ax.tick_params(colors=MUTED)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.suptitle(f"Current boundary vs. reference: {name} (gray: hidden columns) — "
                 f"RMS error boundary.py {np.sqrt((e_cur ** 2).mean()):.2e}, "
                 f"baseline {np.sqrt((e_base ** 2).mean()):.2e}", fontsize=13, color=TEXT)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"written {out}")


rho_ref, u_ref = P.reference()
rho, u = run(make_outlet)
rho_base, u_base = run(lambda f: lt.EquilibriumOutletP(direction=[1, 0], flow=f, rho_outlet=1.0))

out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "solution.png"
# Boundary columns are not part of the metric; the inlet column also holds
# populations streamed in periodically from the outlet. The vorticity at
# x = 1 uses x = 0 in its finite difference, so it is hidden as well.
plot(rho_ref - 1, rho - 1, rho_base - 1, [0, -1], "density", "ρ − 1", "ρ − ρ_ref", out)
plot(vorticity(u_ref), vorticity(u), vorticity(u_base), [0, 1, -1], "vorticity", "ω", "ω − ω_ref",
     out.with_name(f"{out.stem}_vorticity{out.suffix}"))
