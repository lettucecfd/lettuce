import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
import torch

"""
For descriptions of the initialization, refer to
'01b_first_example_obstacle.py'.

Here, we use a lower Reynolds number for faster convergence.
"""

lattice = lt.Lattice(lt.D2Q9, device="cpu", dtype=torch.float32)
nx = 100
ny = 100
Re = 10
Ma = 0.1
ly = 1

flow = lt.Obstacle((nx, ny), reynolds_number=Re, mach_number=Ma,
                   lattice=lattice, domain_length_x=ly)

x, y = flow.grid
r = .05     # radius
x_c = 0.3   # center along x
y_c = 0.5   # center along y
flow.mask = ((x - x_c) ** 2 + (y - y_c) ** 2) < (r ** 2)

collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)

streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision,
                           streaming=streaming)

simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=100,
                                           filename_base="./output"))

"""
We now add a reporter which we access later. The output can be written to files
specified by out="reporter.txt"
"""
energy = lt.IncompressibleKineticEnergy(lattice, flow)
simulation.reporters.append(lt.ObservableReporter(energy, interval=1000,
                                                  out=None))

"""
Now, we do not just run the whole simulation for 30,000 steps, but check the
energy convergence every 2000 steps.
The populations are kept on the GPU until evaluated by [...].cpu()
"""
nmax = 30000
ntest = 2000
simulation.initialize_f_neq()
it = 0
i = 0
mlups = 0
energy_old = 1
energy_new = 1
while it <= nmax:
    i += 1
    it += ntest
    mlups += simulation.step(ntest)
    energy_new = energy(simulation.f).cpu().mean().item()
    print(f"avg MLUPS: {mlups / i:.3f}, avg energy: {energy_new:.8f}, "
          f"rel. diff: {abs(energy_new - energy_old)/energy_old:.8f}")
    if not energy_new == energy_new:
        print("CRASHED!")
        break
    if abs(energy_new - energy_old)/energy_old < 0.0075:

        print(f"CONVERGED! To {abs(energy_new - energy_old)/energy_old:.2%} "
              f"after {it} iterations.")
        break
    energy_old = energy_new
    u = flow.units.convert_velocity_to_pu(
        lattice.u(simulation.f)).cpu().numpy()
    u_norm = np.linalg.norm(u, axis=0).transpose()
    plt.imshow(u_norm)
    plt.title(f'Velocities at it={it}')
    plt.show()
