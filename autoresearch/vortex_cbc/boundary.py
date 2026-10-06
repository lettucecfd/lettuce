"""
Candidate outflow boundary. THIS IS THE FILE THE AGENT EDITS.

`make_outlet(flow)` must return a lettuce Boundary for the right edge of the
domain (x = -1). It runs after collision and before streaming; see
lettuce/_flow.py (Boundary) and lettuce/ext/_boundary/ for the interface and
existing examples.

Baseline: lettuce's EquilibriumOutletP. It imposes a constant density at the
outlet and copies the velocity from the neighbouring node, so it reflects
pressure waves almost completely.
"""
import lettuce as lt


def make_outlet(flow):
    return lt.EquilibriumOutletP(direction=[1, 0], flow=flow, rho_outlet=1.0)
