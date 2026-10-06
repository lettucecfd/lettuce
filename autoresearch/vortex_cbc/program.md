# autoresearch: characteristic outflow boundary for a convected vortex

You are an autonomous researcher. Your goal is an outflow boundary condition
for lettuce that lets a vortex and the acoustic waves it carries leave the
domain **without reflection**: a non-reflecting, characteristic boundary
condition (CBC).

## The experiment

A Lamb-Oseen vortex is convected by a uniform flow (Ma = 0.1, Re = 500,
D2Q9, BGK) through the right edge of a 100 x 100 domain, periodic in y, with a
fixed equilibrium inflow on the left. After 2000 steps the vortex has left.
The result is compared with a reference solution on a periodic domain long
enough that nothing ever comes back, i.e. what an infinite domain would give.

- `prepare.py` — fixed constants, flow setup, reference solution, metric. **Do not modify.**
- `evaluate.py` — runs the evaluation and prints the result. **Do not modify.**
- `boundary.py` — the candidate boundary. **This is the only file you edit.**

The metric is `score` (lower is better): the mean of the RMS density error and
the RMS velocity error against the reference, normalised by the vortex
amplitude. `err_rho_late` is the density error after the vortex has left — it
isolates reflected acoustic waves and is a useful diagnostic. The baseline
(`EquilibriumOutletP`) scores about 0.047.

## Rules

1. Edit only `boundary.py`. You may add helper classes and functions there.
   You may not change anything else, add dependencies, or alter lettuce itself.
2. The boundary acts only on the outlet column (x = -1). It may read the flow
   state (`f`, density, velocity) on that column and on interior neighbours,
   keep its own state between time steps (e.g. previous values for time
   derivatives) and use `flow.units`, `flow.stencil`, `flow.equilibrium`.
3. No knowledge of the test case beyond what a real outflow boundary would
   have: do not import from `prepare.py`, do not read the reference cache,
   do not hard-code the vortex, its position or its parameters. Reasonable
   inputs are the far-field state a user would prescribe (density 1, the mean
   inflow velocity from `flow.units`) and tuning constants.
4. Every evaluation must finish within the time limit (300 s); a run that
   diverges or times out counts as a failure.
5. Simplicity matters. A tiny improvement that doubles the code is usually
   not worth keeping; removing code at equal score is a win.

## Setup

1. Agree on a run tag with the user (e.g. `oct06`) and create the branch
   `autoresearch/vortex-cbc-<tag>` from the current branch.
2. Read `prepare.py`, `evaluate.py`, `boundary.py`, the `Boundary` interface in
   `lettuce/_flow.py`, the existing boundaries in `lettuce/ext/_boundary/` and
   how `lettuce/_simulation.py` applies them (post-collision, before
   streaming; `make_no_collision_mask` / `make_no_streaming_mask`).
3. Make sure the reference exists: `uv run --extra cpu python prepare.py`
   (runs once, about 30 s on a CPU).
4. Create `results.tsv` with the header line (tab separated):
   `commit	score	err_rho_late	status	description`
5. Run the baseline unchanged and record it as the first row.

## The experiment loop

Run from `autoresearch/vortex_cbc/`. Loop forever:

1. Look at the current state: the branch, the last kept commit, `results.tsv`.
2. Change `boundary.py` with one experimental idea.
3. `git commit` the change.
4. Run `uv run --extra cpu python evaluate.py > run.log 2>&1` (redirect
   everything; do not let output flood your context).
5. Read the result: `grep -E "^(status|score|err_rho_late):" run.log`. If
   nothing matches, the run crashed: `tail -n 50 run.log`, fix obvious
   mistakes and retry; give up on the idea after a few attempts.
6. Append a row to `results.tsv` (short commit hash, score, err_rho_late,
   `keep` / `discard` / `crash`, what you tried). Do not commit
   `results.tsv`; it stays untracked.
7. If the score improved, keep the commit and advance. Otherwise reset to the
   last kept commit (`git reset --hard <commit>`).

**Never stop to ask whether to continue.** The user may be away and expects
you to keep working until interrupted. When you run out of ideas, think
harder: reread the papers below, combine near misses, try more radical
changes, revisit discarded ideas with different parameters.

## Starting points

Characteristic boundary conditions decompose the flow at the boundary into
waves travelling in and out (LODI, Thompson 1987; Poinsot & Lele 1992) and
suppress the incoming ones, usually with a relaxation towards the far-field
pressure (factor sigma, Rudy & Strikwerda 1980). For lattice Boltzmann, see:

- Izquierdo & Fueyo, "Characteristic nonreflecting boundary conditions for
  open boundaries in lattice Boltzmann methods", Phys. Rev. E 78 (2008)
- Heubes, Bartel & Ehrhardt, "Characteristic boundary conditions in the
  lattice Boltzmann method for fluid and gas dynamics", J. Comput. Appl.
  Math. 262 (2014)
- Jung, Sagaut et al., "Non-reflecting boundary conditions for the lattice
  Boltzmann method", J. Comput. Phys. 2015
- Wissocq, Gourdain, Malaspinas & Eyssartier, "Regularized characteristic
  boundary conditions for the Lattice-Boltzmann methods at high Reynolds
  number flows", J. Comput. Phys. 331 (2017) — uses this very test case

Typical building blocks: estimating macroscopic derivatives at the boundary
by one-sided finite differences, solving the LODI system for the boundary
values, reconstructing the unknown populations (equilibrium plus
non-equilibrium, regularised), and treating transverse terms.
