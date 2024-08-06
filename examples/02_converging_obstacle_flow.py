import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
import torch

"""
For descriptions of the initialization, refer to
'01b_first_example_obstacle.py'.

Here, we use a lower Reynolds number for faster convergence.
"""

context = lt.Context(device=torch.device('cuda:0' if torch.cuda.is_available()
                     else 'cpu'), dtype=torch.float32, use_native=False)
nx = 300
ny = 100
Re = 10
Ma = 0.1
lx = 1

flow = lt.Obstacle(context, [nx, ny], reynolds_number=Re, mach_number=Ma,
                   domain_length_x=lx)

x, y = flow.grid
r = .05*y.max()     # radius
x_c = 0.3*x.max()   # center along x
y_c = 0.5*y.max()   # center along y
flow.mask = ((x - x_c) ** 2 + (y - y_c) ** 2) < (r ** 2)

collision = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu)
simulation = lt.Simulation(flow=flow, collision=collision, reporter=[])

simulation.reporter.append(lt.VTKReporter(interval=1000,
                                          filename_base=
                                          "./data/converging_obstacle"))

"""
We now add a reporter which we access later. The output can be written to files
specified by out="reporter.txt"
"""
energy = lt.IncompressibleKineticEnergy(flow)
simulation.reporter.append(lt.ObservableReporter(energy, interval=1000,
                                                 out=None))

"""
Now, we do not just run the whole simulation for 30,000 steps, but check the
energy convergence every 2000 steps.
The populations are kept on the GPU until evaluated by [...].cpu()
"""
nmax = 30000
ntest = 1000
it = 0
i = 0
mlups = 0
energy_old = 1
energy_new = 1
while it <= nmax:
    i += 1
    it += ntest
    mlups += simulation(num_steps=ntest)
    energy_new = flow.incompressible_energy().cpu().mean().item()
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
    u = context.convert_to_ndarray(flow.u_pu)
    u_norm = np.linalg.norm(u, axis=0).transpose()
    plt.imshow(u_norm)
    plt.colorbar()
    plt.title(f'Velocities at it={it}')
    plt.show()
