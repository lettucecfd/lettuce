# Vortex outflow: autoresearch on characteristic boundary conditions

An AI agent iteratively improves a non-reflecting outflow boundary condition
for lettuce, following the setup of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch): a fixed
evaluation, one file the agent edits, and a loop of experiment → measure →
keep or discard.

The test case is a Lamb-Oseen vortex convected through the outlet of a 2D
domain, scored against a reference on an effectively infinite domain
(Wissocq et al. 2017).

| File | Role |
|---|---|
| `program.md` | Instructions for the agent: rules, setup, experiment loop |
| `prepare.py` | Fixed: test case, reference solution, metric |
| `evaluate.py` | Fixed: scores `boundary.py` and prints the result |
| `boundary.py` | Edited by the agent: the candidate outflow boundary |

## Running

```console
cd autoresearch/vortex_cbc
uv run --extra cpu python prepare.py     # reference solution, once (~30 s)
uv run --extra cpu python evaluate.py    # score boundary.py (~2 s)
```

To start a research run, open an agent (e.g. Claude Code) in this directory
and tell it: *"Read program.md and start the experiment loop."*

Baseline (`EquilibriumOutletP`): `score ≈ 0.047`, `err_rho_late ≈ 0.006`.
