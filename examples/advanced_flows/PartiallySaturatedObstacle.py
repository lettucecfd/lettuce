import torch
import os
from matplotlib import pyplot as plt
from matplotlib.colors import colorConverter, LinearSegmentedColormap
import lettuce as lt
import numpy as np

"""
Setting up variable parameters
"""
dim = 2
context = lt.Context(use_native=False)
Re = 250
Ma = 0.01
saturation = 0.5

"""
Setting up the Flow object
"""


class ObstaclePartially(lt.Obstacle):
    def __init__(self, resolution, domain_length_x):
        super().__init__(context, resolution, Re, Ma, domain_length_x)
        self.saturation = saturation
        return

    @property
    def boundaries(self):
        x = self.grid[0]
        return [
            lt.EquilibriumBoundaryPU(
                context,
                torch.abs(x) < 1e-6,
                self.units.characteristic_velocity_pu * self._unit_vector(),
                pressure=0
            ),
            lt.EquilibriumOutletP(
                self._unit_vector().tolist(),
                self
            ),
            lt.PartiallySaturatedBC(
                self.mask,
                tau=self.units.relaxation_parameter_lu,
                saturation=self.saturation)
        ]

    def initial_solution(self, x):
        p = np.zeros_like(x[0], dtype=float)[None, ...]
        ui = np.zeros_like(x[0], dtype=float)
        u = np.stack((ui, ui) if self.units.lattice.D == 2 else (ui, ui, ui))
        return p, u


"""
Setting up post-processing
"""


class Show2D:
    def __init__(self, mask, outdir: str, **kwargs):
        if len(mask.shape) == 3:
            mask = mask[:, :, int(mask.shape[2] / 2)]
        self.mask = context.convert_to_ndarray(mask).transpose()
        self.outdir = outdir

        self.dpi = kwargs['dpi'] if 'dpi' in kwargs else 1200
        self.save = kwargs['save'] if 'save' in kwargs else True
        self.show_mask = kwargs['show_mask'] if 'show_mask' in kwargs else True
        self.mask_alpha = kwargs['mask_alpha'] if 'mask_alpha' in kwargs \
            else 1.

        self.__call__(mask, "solid_mask", "solid_mask")

    def __call__(self, data, title: str, name: str, vlim=None):
        fig, ax = plt.subplots(figsize=(16, 4))
        data = data[:, :, int(data.shape[2] / 2)] if len(data.shape) > 2 \
            else data
        data = context.convert_to_ndarray(data).transpose() \
            if isinstance(data, torch.Tensor) else data.transpose()
        vmin, vmax = vlim if vlim is not None else None, None
        p = ax.imshow(data, origin='lower', aspect='auto',
                      interpolation='none', vmin=vmin, vmax=vmax)
        color1 = colorConverter.to_rgba('white')
        color2 = colorConverter.to_rgba('black')
        cmap2 = LinearSegmentedColormap.from_list('my_cmap2',
                                                  [color1, color2], 256)
        cmap2._init()
        alphas = np.linspace(0, 0.8, cmap2.N + 3)
        cmap2._lut[:, -1] = alphas
        if self.show_mask:
            ax.imshow(self.mask, origin='lower', aspect='auto',
                      interpolation='none', cmap=cmap2, vmin=0, vmax=1,
                      alpha=self.mask_alpha)
        ax.set_title(title)
        fig.colorbar(p, ax=ax)
        plt.show()
        if self.save:
            fig.savefig(os.path.join(self.outdir, name), dpi=self.dpi)
        plt.close()


"""
initializing up Flow and Show2D
"""
nz = 1 if dim == 2 else 10
ny = 50
nx = 2 * ny
flow = ObstaclePartially(
    resolution=[nx, ny, nz] if dim == 3 else [nx, ny],
    domain_length_x=10.1
)
x, y, *z = flow.grid
# flow.mask = ((x >= 2) & (x < 5) & (y >= x) & (y <= 3))  # triangle
flow.mask = (x >= 2) & (x <= 4) & (y >= 1.5) & (y <= 3.5)  # rectangle

collision = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu)
simulation = lt.Simulation(flow=flow, collision=collision, reporter=[])
imgdir = os.path.join(os.getcwd(), "images")
if not os.path.exists(imgdir):
    os.mkdir(imgdir)
if not os.path.exists('data'):
    os.mkdir('data')
simulation.reporter.append(lt.VTKReporter(
    interval=200,
    filename_base="data/partiallysaturated/out"
))
show2d = Show2D(flow.mask, imgdir, save=False, show_mask=False,
                mask_alpha=saturation)
show2d(flow.u_pu[0], "u_x(t=0)", "u_0", vlim=(-.2, 1.5))


"""
run simulation
"""
steps, mlups_sum, n = 0, 0, 0
step_size = 1000
t_max = 20
step_max = min(flow.units.convert_time_to_lu(t_max), 20000)
print(f"Step size: {step_size}, max steps: {step_max:.1f}, max time: {t_max}")
while steps < step_max and ~torch.isnan(flow.f).any():
    mlups_new = simulation(step_size)
    steps += step_size
    n += 1
    mlups_sum += mlups_new
    mlups_avg = mlups_sum / n
    energy = (torch.sum(flow.incompressible_energy()) /
              ((1 / nz) ** dim))
    t = flow.units.convert_time_to_pu(steps)
    print(f"Step: {steps}, Time: {t:.1f} s, Energy: {energy:.2f}, "
          f"MLUPS: {mlups_avg:.1f}")
    u = flow.u_pu
    p = flow.p_pu  # [Pa]
    grad_u0 = torch.gradient(u[0])
    grad_u1 = torch.gradient(u[1])
    vorticity = (grad_u1[0] - grad_u0[1])
    show2d(torch.abs(vorticity), f"vorticity(it={steps},t={t:.1f})",
           f"vort_{steps}", vlim=(-.2, 1))
    show2d(p * 10 ** -5, f"pressure(it={steps},t={t:.1f}) [bar]",
           f"rho_{steps}", vlim=(-.001, .001))
    show2d(torch.norm(u, dim=0), f"u(it={steps},t={t:.1f}) [m/s]",
           f"u_{steps}", vlim=(-.2, 1.5))
