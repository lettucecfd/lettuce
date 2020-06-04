import lettuce as lt
import torch
import numpy as np
import matplotlib.pyplot as plt

print("start")

# ---------- Set up simulation -------------
device = torch.device("cuda:0")  # replace with CPU, if no GPU is available
lattice = lt.Lattice(lt.D3Q27, device=device, dtype=torch.float64)
resolution = 100  # resolution of the lattice, low resolution leads to unstable speeds somewhen after 10 (PU)
flow = lt.TaylorGreenVortex3D(resolution, 1600, 0.1, lattice)
collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow, lattice, collision, streaming)

# reporters will grab the results in between simulation steps (see io.py and simulation.py)
# Output: Column 1: time in LU, Column 2: kinetic energy in PU
kinE_reporter = lt.EnergyReporter(lattice, flow, interval=1, out=None)
simulation.reporters.append(kinE_reporter)
# Output: separate VTK-file with ux,uy,uz and p for every time step in ../data
VTKreport = lt.VTKReporter(lattice, flow, interval=5)
simulation.reporters.append(VTKreport)

# ---------- Simulate until time = 20 (PU) -------------
print("Simulating", int(simulation.flow.units.convert_time_to_lu(10)), "steps! Maybe drink some water in the meantime.")
simulation.step(int(simulation.flow.units.convert_time_to_lu(10)))

# ---------- Plot kinetic energy over time (PU) -------------
# grab output of kinetic energy reporter
E = np.asarray(kinE_reporter.out)
# normalize to size of grid, not always necessary
E[:, 1] = E[:, 1] / (2*np.pi)**3
# save kinetic energy values for later use
np.save("TGV3DoutRes" + str(resolution) + "E", E)
fig = plt.figure()
ax1 = plt.subplot(1, 2, 1)
ax1.plot(simulation.flow.units.convert_time_to_pu(range(0, E.shape[0])), E[:, 1])

# ---------- Plot magnitude of speed in slice of 3D volume -------------
# grab u in PU
u = simulation.flow.units.convert_velocity_to_pu(simulation.lattice.u(simulation.f))
# [direction of u: Y, X, Z] (due to ij indexing)
uMagnitude = torch.pow(torch.pow(u[0, :, :, :], 2) + torch.pow(u[1, :, :, :], 2) + torch.pow(u[2, :, :, :], 2), 0.5)
# select slice to plot
uMagnitude = uMagnitude[:, :, round(0.1 * resolution)]
# send selected slice to CPU und numpy, to be able to plot it via matplotlib
uMagnitude = uMagnitude.cpu().numpy()
ax2 = plt.subplot(1, 2, 2)
ax2.matshow(uMagnitude)
plt.show()
