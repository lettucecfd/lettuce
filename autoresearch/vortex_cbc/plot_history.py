"""
Plot how the solution improved over a research run: every committed version
of boundary.py on the current branch (baseline included) is simulated again,
and its density and vorticity error over time is drawn as one line.
Writes history.png.

    uv run --extra cpu python plot_history.py [history.png]
"""
import subprocess
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import prepare as P

HERE = Path(__file__).parent
# categorical order, oldest version first; later versions are folded into gray
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
STEPS = (np.arange(P.NUM_STEPS // P.SAMPLE_EVERY) + 1) * P.SAMPLE_EVERY


def versions():
    """(short hash, subject) of every commit that changed boundary.py, oldest first."""
    log = subprocess.run(["git", "log", "--reverse", "--format=%h %s", "--", "boundary.py"],
                         cwd=HERE, capture_output=True, text=True, check=True).stdout
    return [line.split(" ", 1) for line in log.splitlines()]


def load(commit):
    """Import boundary.py as it was in `commit`."""
    source = subprocess.run(["git", "show", f"{commit}:./boundary.py"], cwd=HERE,
                            capture_output=True, text=True, check=True).stdout
    module = types.ModuleType(f"boundary_{commit}")
    exec(compile(source, f"boundary.py@{commit}", "exec"), module.__dict__)
    return module.make_outlet


def vorticity(u):
    return np.gradient(u[:, 1], axis=1) - np.gradient(u[:, 0], axis=2)


def errors(rho, u, rho_ref, u_ref):
    """RMS density and vorticity error over time, normalised by the initial peak.
    Hidden as in plot_solution.py: boundary columns, and x = 1 for vorticity."""
    w, w_ref = vorticity(u), vorticity(u_ref)
    e_rho = np.sqrt(((rho - rho_ref)[:, 1:-1] ** 2).mean(axis=(1, 2)))
    e_w = np.sqrt(((w - w_ref)[:, 2:-1] ** 2).mean(axis=(1, 2)))
    return e_rho / np.abs(rho_ref[0] - 1).max(), e_w / np.abs(w_ref[0]).max()


rho_ref, u_ref = P.reference()
runs = []
for commit, subject in versions():
    flow = P.ConvectedVortex(P.make_context(), P.NX, P.NX // 2,
                             boundary_factory=lambda f, m=load(commit): [P.inlet(f), m(f)])
    result, status = P._run(flow, slice(0, P.NX))
    score = P.evaluate(load(commit))["score"] if result is not None else float("inf")
    print(f"{commit}  score {score:.6f}  {subject}")
    runs.append((commit, subject, score, errors(*result, rho_ref, u_ref) if result else None))

fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
for k, (ax, name) in enumerate(zip(axes, ["density", "vorticity"])):
    for i, (commit, subject, score, err) in enumerate(runs):
        if err is None:
            continue
        color = COLORS[i] if i < len(COLORS) else "#b9b8b2"
        label = f"{i}: {commit}  score {score:.2e}  {subject[:55]}" if i < len(COLORS) else None
        ax.plot(STEPS, err[k], color=color, lw=2, label=label)
    ax.set_yscale("log")
    ax.set_ylabel(f"RMS {name} error\n/ initial peak", color=TEXT)
    ax.tick_params(colors=MUTED)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].legend(frameon=False, fontsize=8.5, loc="lower center")
axes[1].set_xlabel("time step", color=TEXT)
fig.suptitle(f"Error over time for every committed boundary.py ({len(runs)} versions, "
             "oldest = 0)", fontsize=13, color=TEXT)
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "history.png"
fig.savefig(out, dpi=100)
print(f"written {out}")
