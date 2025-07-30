import torch
import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

"""
Context definitions.

The context defines the default device (cpu or cuda) and datatype (e.g.,
float32 for single, float64 for double precision).
Native CUDA is currently not supported for the anti-bounce-back outlet.
"""
context = lt.Context(torch.device("cuda:0"), use_native=False)
np.random.seed(0)
"""
Flow definitions.

We need
1. the resolution in x and y direction
2. the Reynolds number (i.e., how fast the flow behaves compared to the
    object's length and fluid's viscosity)
3. the Mach number (i.e., how fast the flow is compared to speed of sound;
    Ma=0.3 is stable, above is discouraged)
4. the physical domain length in x-direction (this defines how lattice units
    scale to physical units)
to initialize the Obstacle flow object.
"""
nx = 100
ny = 100
Re = 6000
Ma = 0.1
lx = 1

flow = lt.Obstacle(context, [nx, ny], reynolds_number=Re, mach_number=Ma,
                   domain_length_x=lx)

"""
Per default, lt.Obstacle has no solid. It is stored in flow.mask as a fully
False numpy array.
To add a solid, we set some mask values to True by getting the domain extends
from flow.grid and creating a boolean array from it.

For a circle, just use a boolean function. Otherwise, you may as well use the
array indices.
"""
x, y = flow.grid
r = .05*y.max()      # radius
x_c = 0.3*x.max()   # center along x
y_c = 0.5*y.max()   # center along y
mask = np.bool(x)
mask[:,:] = False
mask[10:30,45:55] = True

flow.mask = ((x - x_c) ** 2 + (y - y_c) ** 2) < (r ** 2)
flow.mask = mask

"""
To show 2D images, you need to rotate the outputs. This is because in lettuce,
the first axis is downstream, while for imshow it is vertical.
"""
plt.imshow(context.convert_to_ndarray(flow.mask.t()))
plt.show()

"""
Collision definition.

The collision is usually BGK (low dissipation, but may be unstable) or KBC
(higher dissipation, but generally stable). BGK is preferred for converging
flows, KBC is preferred for driven flows in smaller domains (where energy
conversation plays a smaller role, but gradients may be higher).
"""
collision = lt.KBCCollision2D(tau=flow.units.relaxation_parameter_lu)

"""
Simulation object setup.
"""
simulation = lt.Simulation(flow=flow, collision=collision, reporter=[])

"""
Reporters.

- Reporter objects are used to extract information later on or during the
    simulation.
- They can be created as separate objects when required later
    (see 01_example4convergence.py).
"""

energyreporter = lt.ObservableReporter(lt.IncompressibleKineticEnergy(flow),
                                       interval=50)
simulation.reporter.append(energyreporter)

"""
Run num_steps iterations of the simulation.
This can be done repeatedly (see 02_converging_flows.py).
"""
mlups = simulation(num_steps=8700)
mlups = simulation(num_steps=1)
print("Performance in MLUPS:", mlups)

"""
Before or after simulation call, physical values can be extracted from the
flow.
Alternatively, the reporters can be drawn from the simulation.reporters list
(see 01_example4convergence.py)
"""

u = context.convert_to_ndarray(flow.u_pu)
u_norm = np.linalg.norm(u, axis=0).transpose()
fig, ax = plt.subplots()

ax.imshow(u_norm, cmap="gray")
ax.set_xticks([])
ax.set_yticks([])

# Define the rectangle properties
rect_y = 45  # Starting x-coordinate
rect_x = 10  # Starting y-coordinate
rect_width = 20  # Width of the rectangle
rect_height = 10  # Height of the rectangle

# Create a Rectangle patch and add it to the plot
rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                 linewidth=1, edgecolor='black', facecolor='white', hatch="///")
ax.add_patch(rect)

# plt.title('Velocity after simulation with rectangle mask')
# plt.colorbar()
plt.savefig("Rectangle.pdf",
            format='pdf',
            bbox_inches='tight',
            pad_inches=0.01,
            transparent=False)
plt.tight_layout()
plt.show()
