"""
This file showcases the simplicity of the lettuce code.
The following code will run a two-dimensional Taylor-Green vortex on GPU.
"""

import torch
import lettuce as lt

flow = lt.TaylorGreenVortex(
    lt.Context(dtype=torch.float64),  # for running on cpu: device='cpu'
    resolution=128,
    reynolds_number=100,
    mach_number=0.05,
    stencil=lt.D2Q9
)
simulation = lt.Simulation(
    flow=flow,
    collision=lt.BGKCollision(tau=flow.units.relaxation_parameter_lu),
    reporter=[])
mlups = simulation(num_steps=1000)
print("Performance in MLUPS:", mlups)
