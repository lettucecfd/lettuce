"""
Lamb-Oseen Vortex in 2D.

This module defines the `LambOseenVortex2D` class, which simulates a 2D Lamb-Oseen vortex
flow according to Wissocq et al. 2017
"""
import warnings
from typing import Union, List, Optional

import torch

from ... import UnitConversion
from .._stencil import D2Q9
from . import ExtFlow

__all__ = ['LambOseenVortex2D']


class LambOseenVortex2D(ExtFlow):
    def __init__(self, context: 'Context', resolution: Union[int, List[int]],
                 reynolds_number, mach_number,
                 stencil: Optional['Stencil'] = None,
                 equilibrium: Optional['Equilibrium'] = None,
                 initialize_fneq: bool = True,
                 velocity_init = 1,
                 K = None,
                 xc: int = None):
        # Store configuration parameters
        self.initialize_fneq = initialize_fneq
        self.velocity_init = velocity_init

        # Set default stencil if not provided
        if stencil is None and not isinstance(resolution, list):
            self.stencil = D2Q9()
        else:
            self.stencil = stencil() if callable(stencil) else stencil
        # Set the vortex center x-coordinate.
        if isinstance(resolution, int):
            self.xc = resolution // 2 if xc is None else xc
        else:
            self.xc = resolution[0] // 2 if xc is None else xc


        # Call the base class constructor. This also sets self.context, self.resolution,
        # self.units, and self.equilibrium.
        ExtFlow.__init__(self, context, resolution, reynolds_number,
                         mach_number, self.stencil, equilibrium)


    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * self.stencil.d
        else:
            assert len(resolution) == 2, 'expected 2-dimensional resolution'
            return resolution

    def make_units(self, reynolds_number, mach_number,
                   resolution: List[int]) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution[0], characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    @property
    def grid(self):
        endpoints = self.resolution
        xyz = tuple([
            self.units.convert_length_to_pu(torch.arange(0, endpoints[0], device=self.context.device, dtype=self.context.dtype)),
            self.units.convert_length_to_pu(torch.arange(0, endpoints[1], device=self.context.device, dtype=self.context.dtype))
        ])
        return torch.meshgrid(*xyz, indexing='ij')

    def initial_pu(self) -> (torch.Tensor, torch.Tensor):
        p, U = self.initial_lamboseenvortex()
        return p, U

    def initial_lamboseenvortex(self) -> (torch.Tensor, torch.Tensor):
        # Vortex center coordinates in lattice units
        # xc is already in lattice units from __init__
        yc = self.resolution[1] * 0.5  # y-center in lattice units

        # Get grid coordinates in physical units and then convert to lattice units for calculation
        x_pu, y_pu = self.grid  # x_pu, y_pu are (nx, ny) in physical units

        # Convert grid coordinates to lattice units for calculation with Rc (which is a characteristic length in LU)
        x_lu = self.units.convert_length_to_lu(x_pu)
        y_lu = self.units.convert_length_to_lu(y_pu)

        # Initial characteristic velocity in lattice units
        ux0_lu = self.units.convert_velocity_to_lu(self.velocity_init)

        # Lamb-Oseen vortex specific constants (these might be configurable in a more advanced version)
        beta = 0.5  # Parameter related to circulation
        Rc = 20.0  # Characteristic radius of the vortex core in lattice units
        gamma = 0.5  # Adiabatic index or similar thermodynamic constant
        Cv = 1.0 / 3.0  # Specific heat capacity at constant volume or similar constant

        # Calculate the squared distance from the vortex center in lattice units
        r2_lu = (x_lu - self.xc) ** 2 + (y_lu - yc) ** 2

        # Calculate density (d) based on the thermodynamic relation for the vortex
        # This formula describes the density perturbation due to the vortex.
        d_lu = torch.pow(
            1.0 - (beta * ux0_lu) ** 2 / (2.0 * Cv) * torch.exp(1.0 - r2_lu / (2.0 * Rc)),
            1.0 / (gamma - 1.0)
        )

        # Calculate the exponential decay term for velocity
        exp_term = torch.exp(-r2_lu / (2.0 * Rc))

        # Calculate velocity components (u_x, u_y) in lattice units
        # These are derived from the stream function of the Lamb-Oseen vortex
        u_x_lu = (ux0_lu - beta * ux0_lu * (y_lu - yc) / Rc * exp_term)
        u_y_lu = beta * ux0_lu * (x_lu - self.xc) / Rc * exp_term

        # Convert density (d) to pressure (p) in physical units
        # Assuming 'd' here acts like a density in LU that can be directly converted to pressure in PU.
        # This might depend on the specific LBM pressure definition.
        p_pu = self.units.convert_density_lu_to_pressure_pu(d_lu)

        # Convert velocity components from lattice units to physical units
        u_x_pu = self.units.convert_velocity_to_pu(u_x_lu)
        u_y_pu = self.units.convert_velocity_to_pu(u_y_lu)

        # Stack velocity components into a single tensor (2, nx, ny)
        U_pu = torch.stack([u_x_pu, u_y_pu], dim=0)

        return p_pu, U_pu

    @property
    def post_boundaries(self) -> List['Boundary']:
        return []
