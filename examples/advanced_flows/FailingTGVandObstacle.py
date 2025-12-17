"""
This file showcases:
 a) interrupting a TGV simulation using a reporter that detects NaN in f
 b) interrupting an obstacle simulation using a reporter that detects Ma > 0.3
"""

import torch
import lettuce as lt
import os

if not os.path.exists("./data"):
    os.mkdir("./data")

# a) unstable TGV, that causes NaN values in f, which are detected by the NaN
# ... reporter, who interrupts the simulation.
flow = lt.TaylorGreenVortex(
    lt.Context(dtype=torch.float64),
    resolution=5,
    reynolds_number=30000,
    mach_number=0.3,
    stencil=lt.D2Q9
)
nan_reporter = lt.NaNReporter(100, outdir="./data/nan_reporter", vtk_out=True)
simulation = lt.BreakableSimulation(
    flow=flow,
    collision=lt.BGKCollision(tau=flow.units.relaxation_parameter_lu),
    reporter=[nan_reporter])
simulation(10000)
print(f"Failed after {nan_reporter.failed_iteration} iterations")

############################################
# b) unstable obstacle flow,m that causes high Ma values (Ma > 0.3), which are
# ... detected by the HighMa reporter, who interrupts the simulation.
flow = lt.Obstacle(
    lt.Context(dtype=torch.float64,use_native=False),
    resolution=[32, 32],
    reynolds_number=100,
    mach_number=0.01,
    stencil=lt.D2Q9(),
    domain_length_x=32
)
flow.mask = ((2 < flow.grid[0]) & (flow.grid[0] < 10)
             & (2 < flow.grid[1]) & (flow.grid[1] < 10))
high_ma_reporter = lt.HighMaReporter(100, outdir="./data/ma_reporter",
                                     vtk_out=True)
simulation = lt.BreakableSimulation(
    flow=flow,
    collision=lt.BGKCollision(tau=flow.units.relaxation_parameter_lu),
    reporter=[high_ma_reporter])
simulation(10000)
print(f"Failed after {high_ma_reporter.failed_iteration} iterations")
