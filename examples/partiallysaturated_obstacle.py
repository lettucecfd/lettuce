from datetime import datetime
import torch
import os
from matplotlib import pyplot as plt
from matplotlib.colors import colorConverter, LinearSegmentedColormap
import lettuce as lt
from time import time, sleep
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from math import ceil
from lettuce.boundary import EquilibriumBoundaryPU, EquilibriumOutletP, PartiallySaturatedBC
import numpy as np

dim=2
default_device="cuda:0"

lattice = lt.Lattice(lt.D3Q27 if dim == 3 else lt.D2Q9, device=default_device, use_native=False)
Re = 250
Ma = 0.01
saturation = 0.5


class ObstaclePartially(lt.Obstacle):
    def __init__(self, shape, reynolds_number, mach_number, lattice, domain_length_x, saturation):
        super().__init__(shape, reynolds_number, mach_number, lattice, domain_length_x)
        self.saturation = saturation
        return

    @property
    def boundaries(self):
        x = self.grid[0]
        return [
            EquilibriumBoundaryPU(
                np.abs(x) < 1e-6, self.units.lattice, self.units,
                self.units.characteristic_velocity_pu * self._unit_vector()
            ),
            EquilibriumOutletP(self.units.lattice, self._unit_vector().tolist()),
            PartiallySaturatedBC(self.mask, self.units.lattice, tau=self.units.relaxation_parameter_lu,
                                 saturation=self.saturation)
        ]

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        ui = np.zeros_like(x[0], dtype=float)
        u = np.stack((ui, ui) if self.units.lattice.D == 2 else (ui, ui, ui))
        return p, u


class Show2D:
    def __init__(self, lattice, mask, outdir: str, dpi: int = 1200, save: bool = True, show_mask: bool = True,
                 mask_alpha=1):
        self.lattice = lattice
        self.outdir = outdir
        if len(mask.shape) > 2:
            mask = mask[:, :, int(mask.shape[2]/2)]
        self.mask = self.lattice.convert_to_numpy(mask).transpose() if mask is torch.Tensor else mask.transpose()
        self.dpi = dpi
        self.save = save
        self.show_mask = show_mask
        self.mask_alpha = mask_alpha
        self.__call__(mask, "solid_mask", "solid_mask")

    def __call__(self, data, title: str, name: str, vlim=None):
        fig, ax = plt.subplots(figsize=(16, 4))
        if len(data.shape) > 2:
            data = data[:, :, int(data.shape[2]/2)]
        data = self.lattice.convert_to_numpy(data).transpose() if type(data) == torch.Tensor else data.transpose()
        if vlim is not None:
            vmin, vmax = vlim
            p = ax.imshow(data, origin='lower', aspect='auto', interpolation='none', vmin=vmin, vmax=vmax)
        else:
            p = ax.imshow(data, origin='lower', aspect='auto', interpolation='none')
        color1 = colorConverter.to_rgba('white')
        color2 = colorConverter.to_rgba('black')
        cmap2 = LinearSegmentedColormap.from_list('my_cmap2', [color1, color2], 256)
        cmap2._init()  # create the _lut array, with rgba values
        # create your alpha array and fill the colormap with them.
        # here it is progressive, but you can create whathever you want
        alphas = np.linspace(0, 0.8, cmap2.N + 3)
        cmap2._lut[:, -1] = alphas
        if self.show_mask:
            ax.imshow(self.mask, origin='lower', aspect='auto', interpolation='none', cmap=cmap2, vmin=0, vmax=1,
                      alpha=self.mask_alpha)
        ax.set_title(title)
        fig.colorbar(p, ax=ax)
        plt.show()
        if self.save:
            fig.savefig(os.path.join(self.outdir, name), dpi=self.dpi)
        plt.close()

nz = 1 if dim == 2 else 10
ny = 50
nx = 2 * ny
flow = ObstaclePartially(
    shape=(nx, ny, nz) if dim == 3 else (nx, ny),
    reynolds_number=Re,
    mach_number=Ma,
    lattice=lattice,
    domain_length_x=10.1,
    saturation=saturation
)
x, y = flow.grid
flow.mask = ((x >= 2) & (x < 5) & (y >= x) & (y <= 3))  # triangle
# flow.mask = (x >= 1) & (x <= 4) & (y >= 1) & (y <= 4)  # rectangle

collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
streaming = lt.StandardStreaming(lattice)
simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
vtkdir = os.path.join(os.getcwd(), "vtk")
if not os.path.exists(vtkdir):
    os.mkdir(vtkdir)
imgdir = os.path.join(os.getcwd(), "images")
if not os.path.exists(imgdir):
    os.mkdir(imgdir)
simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=200, filename_base=vtkdir))
simulation.initialize_f_neq()
show2d = Show2D(lattice, flow.mask, imgdir, save=False, show_mask=False, mask_alpha=saturation)
show2d(flow.units.convert_velocity_to_pu(lattice.u(simulation.f))[0], "u_x(t=0)", "u_0", vlim=(-.2, 1.5))

# run simulation
steps, mlups_sum, n = 0, 0, 0
step_size = 1000
t_max = 20
step_max = min(flow.units.convert_time_to_lu(t_max), 20000)
print(f"Step size: {step_size}, max steps: {step_max:.1f}, max time: {t_max}")
while steps < step_max and ~torch.isnan(simulation.f).any():
    mlups_new = simulation.step(step_size)
    steps += step_size
    n += 1
    mlups_sum += mlups_new
    mlups_avg = mlups_sum / n
    energy = torch.sum(lattice.incompressible_energy(simulation.f)) / ((1 / nz) ** dim)
    t = flow.units.convert_time_to_pu(steps)
    print(f"Step: {steps}, Time: {t:.1f} s, Energy: {energy:.2f}, MLUPS: {mlups_avg:.1f}")
    u = flow.units.convert_velocity_to_pu(lattice.u(simulation.f))
    # rho = flow.units.convert_density_lu_to_pressure_pu(lattice.rho(simulation.f))  # [Pa]
    # grad_u0 = torch.gradient(u[0])
    # grad_u1 = torch.gradient(u[1])
    # vorticity = (grad_u1[0] - grad_u0[1])
    # show2d(torch.abs(vorticity), f"vorticity(it={steps},t={t:.1f})", f"vort_{steps}", vlim=(-.2, 1))
    # show2d(rho * 10 ** -5, f"density(it={steps},t={t:.1f}) [bar]", f"rho_{steps}", vlim=(-.001, .001))
    show2d(torch.norm(u, dim=0), f"u(it={steps},t={t:.1f}) [m/s]", f"u_{steps}", vlim=(-.2, 1.5))
