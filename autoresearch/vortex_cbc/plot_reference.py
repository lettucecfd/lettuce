"""
Plot the reference solution in the measurement window: density deviation and
vorticity at four times. Writes reference.png next to this file.

    uv run --extra cpu python plot_reference.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import prepare as P

rho, u = P.reference()
samples = [0, 24, 49, 74]
drho = rho - 1.0
vorticity = np.gradient(u[:, 1], axis=1) - np.gradient(u[:, 0], axis=2)
lim_rho = np.abs(drho).max()
lim_vort = np.abs(vorticity).max()

fig, ax = plt.subplots(2, len(samples), figsize=(13, 6.6), constrained_layout=True)
for j, t in enumerate(samples):
    img_rho = ax[0, j].imshow(drho[t].T, origin="lower", cmap="RdBu_r",
                              vmin=-lim_rho, vmax=lim_rho)
    img_vort = ax[1, j].imshow(vorticity[t].T, origin="lower", cmap="PuOr_r",
                               vmin=-lim_vort, vmax=lim_vort)
    ax[0, j].set_title(f"step {(t + 1) * P.SAMPLE_EVERY}")
    for k in (0, 1):
        ax[k, j].axvline(P.NX - 1, color="0.3", lw=1, ls="--")
        ax[k, j].set_xticks([0, P.NX // 2, P.NX - 1])
        ax[k, j].set_yticks([0, P.NY // 2, P.NY - 1])
    ax[1, j].set_xlabel("x")
ax[0, 0].set_ylabel("density  ρ − 1\n\ny")
ax[1, 0].set_ylabel("vorticity  ω\n\ny")
fig.colorbar(img_rho, ax=ax[0], shrink=0.85, label="ρ − 1")
fig.colorbar(img_vort, ax=ax[1], shrink=0.85, label="ω  [1 / time step]")
fig.suptitle("Reference solution in the measurement window "
             "(dashed: position of the outlet in the test run)", fontsize=13)

out = Path(__file__).parent / "reference.png"
fig.savefig(out, dpi=110)
print(f"written {out}")
