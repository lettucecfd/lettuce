"""
This file showcases the simplicity of the lettuce code.
The following code will run a two-dimensional Taylor-Green vortex on GPU.
"""

import torch
import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np

context = lt.Context(dtype=torch.float64, use_native=False)
flow = lt.LambOseenVortex2D(
    context,  # for running on cpu: device='cpu'
    resolution=200,
    reynolds_number=1000,
    mach_number=0.05,
    stencil=lt.D2Q9
)
simulation = lt.Simulation(
    flow=flow,
    collision=lt.BGKCollision(tau=flow.units.relaxation_parameter_lu),
    reporter=[])
mlups = simulation(num_steps=100)
print("Performance in MLUPS:", mlups)

p = context.convert_to_ndarray(flow.p_pu)[0].transpose()
plt.imshow(p, origin='lower')
plt.show()

u = np.linalg.norm(context.convert_to_ndarray(flow.u_pu),axis=0).transpose()
plt.imshow(u, origin='lower')
plt.show()