"""
This file showcases the simplicity of the lettuce code.
The following code will run a two-dimensional Taylor-Green vortex on GPU.
"""

import torch
import lettuce as lt
from lettuce.ext._reporter.progress_reporter import ProgressReporter

flow = lt.TaylorGreenVortex(
    lt.Context(dtype=torch.float64, use_native=False),  # for running on cpu: device='cpu'
    resolution=128,
    reynolds_number=100,
    mach_number=0.05,
    stencil=lt.D2Q9
)

progress_reporter = ProgressReporter(interval=10000, t_max=40,  i_target=100000,
                                     print_message=True, outdir="./data/")

simulation = lt.Simulation(
    flow=flow,
    collision=lt.BGKCollision(tau=flow.units.relaxation_parameter_lu),
    reporter=[progress_reporter])
mlups = simulation(num_steps=100000)
print("Performance in MLUPS:", mlups)
