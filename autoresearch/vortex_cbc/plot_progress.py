"""
Plot the progress of a research run from results.tsv: every experiment as a
dot, the best score so far as a step line. Writes progress.png.

    uv run --extra cpu python plot_progress.py [results.tsv] [progress.png]
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
KEEP, DISCARD, TEXT, MUTED, GRID = "#2a78d6", "#b9b8b2", "#0b0b0b", "#52514e", "#e6e5e0"

src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results.tsv"
out = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "progress.png"
rows = list(csv.DictReader(src.open(), delimiter="\t"))
if not rows:
    sys.exit(f"{src} has no experiments yet")

n = range(1, len(rows) + 1)
score = [float(r["score"]) if r["status"] != "crash" else float("nan") for r in rows]
kept = [(i, s, r["description"]) for i, s, r in zip(n, score, rows) if r["status"] == "keep"]
discarded = [(i, s) for i, s, r in zip(n, score, rows) if r["status"] == "discard"]
crashed = [i for i, r in zip(n, rows) if r["status"] == "crash"]

best, best_so_far = float("inf"), []
for s, r in zip(score, rows):
    if r["status"] == "keep":
        best = min(best, s)
    best_so_far.append(best)

fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
ax.step(list(n), best_so_far, where="post", color=KEEP, lw=2, label="best so far")
if discarded:
    ax.scatter(*zip(*discarded), s=36, color=DISCARD, zorder=2, label="discarded")
ax.scatter([k[0] for k in kept], [k[1] for k in kept], s=56, color=KEEP,
           edgecolor="white", linewidth=1.5, zorder=3, label="kept")
if crashed:
    ax.scatter(crashed, [ax.get_ylim()[1]] * len(crashed), marker="x", s=40,
               color=MUTED, zorder=2, label="crashed / diverged", clip_on=False)
# label only the six kept steps with the largest improvement, to stay readable
steps = [(prev[1] - cur[1], cur) for prev, cur in zip(kept, kept[1:])]
for _, (i, s, text) in sorted(steps, reverse=True)[:6]:
    ax.annotate(text[:40], (i, s), xytext=(6, 6), textcoords="offset points",
                fontsize=8, color=TEXT)

ax.set_yscale("log")
ax.set_xlabel("experiment", color=TEXT)
ax.set_ylabel("score (lower is better)", color=TEXT)
ax.tick_params(colors=MUTED)
ax.grid(True, which="both", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="upper right")
ax.set_title(f"{len(rows)} experiments, {len(kept)} kept, best score "
             f"{min(k[1] for k in kept):.4f} (baseline {kept[0][1]:.4f})",
             color=TEXT, loc="left")
fig.savefig(out, dpi=110)
print(f"written {out}")
