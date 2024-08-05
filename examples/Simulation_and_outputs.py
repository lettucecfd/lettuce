import lettuce as lt
import torch
import matplotlib.pyplot as plt

print("start")

# ---------- Set up simulation -------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
context = lt.Context(device=device, dtype=torch.float32)  # single
# precision - float64 for double precision
resolution = 128  # resolution of the lattice, low resolution leads to unstable
# speeds somewhen after 10 (PU)
flow = lt.TaylorGreenVortex(context, resolution, 1600, 0.1, lt.D3Q27)

# select collision model - try also KBCCollision or RegularizedCollision
collision = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu)
simulation = lt.Simulation(flow, collision, [])

# ---------- Simulate until time = 10 (PU) -------------
print("Simulating", int(simulation.flow.units.convert_time_to_lu(10)),
      "steps! Maybe drink some water in the meantime.")
# runs simulation, but also returns overall performance in MLUPS (million
# lattice units per second)
print("MLUPS: ", simulation(int(simulation.flow.units.convert_time_to_lu(10))))

fig = plt.figure()
ax1 = plt.subplot(1, 2, 1)

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
