import lettuce as lt
import torch
import numpy as np
import matplotlib.pyplot as plt

print("start")

# ---------- Set up simulation -------------
device = torch.device("cuda:0") if torch.cuda.is_available() \
    else torch.device("cpu")
context = lt.Context(device=device, dtype=torch.float32)  # single
# precision - float64 for double precision
resolution = 128  # resolution of the lattice, low resolution leads to unstable
# speeds somewhen after 10 (PU)
flow = lt.TaylorGreenVortex3D(context=context, resolution=resolution,
                              reynolds_number=1600, mach_number=0.1)

# select collision model - try also KBCCollision or RegularizedCollision
collision = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu)
simulation = lt.Simulation(flow, collision)

# reporters will grab the results in between simulation steps
# (see io.py and _simulation.py)
# Output: Column 1: time in LU, Column 2: kinetic energy in PU
energy = lt.IncompressibleKineticEnergy(flow)
kinE_reporter = lt.ObservableReporter(energy, out=None)  # output to list
simulation.reporter.append(kinE_reporter)
kinE_reporter2 = lt.ObservableReporter(energy, interval=100)  # console output
simulation.reporter.append(kinE_reporter2)

# Output: separate VTK-file with ux,uy,uz and p for every time step in ../data
VTKReporter = lt.VTKReporter(flow, interval=100)
simulation.reporter.append(VTKReporter)

tend = 10  # time in PU
nsteps = int(simulation.flow.units.convert_time_to_lu(10))  # time in LU
# ---------- Simulate until time = tend (PU) -------------
print(f"Simulating {nsteps} steps! Maybe drink some water in the meantime.")
# runs simulation, but also returns overall performance in MLUPS (million
# lattice units per second)
print(f"MLUPS: {simulation(nsteps)}")

# ---------- Plot kinetic energy over time (PU) -------------
# grab output of kinetic energy reporter
E = np.asarray(kinE_reporter.out)
# normalize to size of grid, not always necessary
E[:, 1] = E[:, 1] / (2 * np.pi) ** 3
# save kinetic energy values for later use
np.save("TGV3DoutRes" + str(resolution) + "E", E)
fig = plt.figure()
ax1 = plt.subplot(1, 2, 1)
plt.xlabel('Time in physical units')
plt.ylabel('Kinetic energy in physical units')
ax1.plot(simulation.flow.units.convert_time_to_pu(range(0, E.shape[0])),
         E[:, 1])

# ---------- Plot magnitude of speed in slice of 3D volume -------------
# grab u in PU
u = flow.u_pu
# [direction of u: Y, X, Z] (due to ij indexing)
uMagnitude = torch.pow(torch.pow(u[0, :, :, :], 2)
                       + torch.pow(u[1, :, :, :], 2)
                       + torch.pow(u[2, :, :, :], 2), 0.5)
# select slice to plot
uMagnitude = uMagnitude[:, :, round(0.1 * resolution)]
# send selected slice to CPU und numpy, to be able to plot it via matplotlib
uMagnitude = uMagnitude.cpu().numpy()
ax2 = plt.subplot(1, 2, 2)
ax2.matshow(uMagnitude)
plt.tight_layout()
plt.show()
plt.savefig('tgv3d-output.pdf')
