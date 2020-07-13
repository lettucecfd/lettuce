"""
Unit conversion into lattice units and back.

lu = lattice units
pu = physical units
"""

import numpy as np
import torch


class UnitConversion:
    """
    Provides unit conversions between physical units (pu) and lattice units (lu).
    Conversion methods should work for floats, torch.tensors and np.arrays.
    """
    def __init__(self, lattice, reynolds_number, mach_number=0.05, characteristic_length_pu=1,
                 characteristic_velocity_pu=1, characteristic_length_lu=1, origin_pu=None,
                 characteristic_density_lu=1, characteristic_density_pu=1):
        self.lattice = lattice
        self.reynolds_number = reynolds_number
        self.mach_number = mach_number
        self.characteristic_length_pu = characteristic_length_pu
        self.characteristic_velocity_pu = characteristic_velocity_pu
        self.characteristic_length_lu = characteristic_length_lu
        self.characteristic_density_lu = characteristic_density_lu
        self.characteristic_density_pu = characteristic_density_pu
        self.origin_pu = np.zeros([lattice.D]) if origin_pu is None else origin_pu

    @property
    def characteristic_velocity_lu(self):
        return self.lattice.stencil.cs * self.mach_number

    @property
    def characteristic_pressure_pu(self):
        return self.characteristic_density_pu * self.characteristic_velocity_pu**2

    @property
    def characteristic_pressure_lu(self):
        return self.characteristic_density_lu * self.characteristic_velocity_lu ** 2

    @property
    def viscosity_lu(self):
        return self.characteristic_length_lu * self.characteristic_velocity_lu / self.reynolds_number

    @property
    def viscosity_pu(self):
        return self.characteristic_length_pu * self.characteristic_velocity_pu / self.reynolds_number

    @property
    def relaxation_parameter_lu(self):
        return self.viscosity_lu / self.lattice.stencil.cs ** 2 + 0.5

    def convert_velocity_to_pu(self, velocity_in_lu):
        return velocity_in_lu / self.characteristic_velocity_lu * self.characteristic_velocity_pu

    def convert_velocity_to_lu(self, velocity_in_pu):
        return velocity_in_pu / self.characteristic_velocity_pu * self.characteristic_velocity_lu

    def convert_acceleration_to_pu(self, acceleration_in_lu):
        return (acceleration_in_lu / (self.characteristic_velocity_lu ** 2 / self.characteristic_length_lu)
                * (self.characteristic_velocity_pu ** 2 / self.characteristic_length_pu))

    def convert_acceleration_to_lu(self, acceleration_in_pu):
        return (acceleration_in_pu / (self.characteristic_velocity_pu ** 2 / self.characteristic_length_pu)
                * (self.characteristic_velocity_lu ** 2 / self.characteristic_length_lu))

    def convert_coordinates_to_pu(self, coordinates_in_lu):
        return (coordinates_in_lu / self.characteristic_length_lu * self.characteristic_length_pu) + self.origin_pu

    def convert_coordinates_to_lu(self, coordinates_in_pu):
        return (coordinates_in_pu - self.origin_pu) / self.characteristic_length_pu * self.characteristic_length_lu

    def convert_time_to_pu(self, time_in_lu):
        return (time_in_lu / (self.characteristic_length_lu/self.characteristic_velocity_lu)
                * (self.characteristic_length_pu/self.characteristic_velocity_pu))

    def convert_time_to_lu(self, time_in_pu):
        return (time_in_pu / (self.characteristic_length_pu/self.characteristic_velocity_pu)
                * (self.characteristic_length_lu/self.characteristic_velocity_lu))

    def convert_density_lu_to_pressure_pu(self, density_lu):
        cs = self.lattice.cs if isinstance(density_lu, torch.Tensor) else self.lattice.stencil.cs
        return self.convert_pressure_to_pu((density_lu-self.characteristic_density_lu) * cs**2)

    def convert_pressure_pu_to_density_lu(self, pressure_pu):
        cs = self.lattice.cs if isinstance(pressure_pu, torch.Tensor) else self.lattice.stencil.cs
        return self.convert_pressure_to_lu(pressure_pu) / cs**2 + self.characteristic_density_lu

    def convert_density_to_pu(self, density_lu):
        return density_lu / self.characteristic_density_lu * self.characteristic_density_pu

    def convert_density_to_lu(self, density_pu):
        return density_pu / self.characteristic_density_pu * self.characteristic_density_lu

    def convert_pressure_to_pu(self, pressure_lu):
        return pressure_lu / self.characteristic_pressure_lu * self.characteristic_pressure_pu

    def convert_pressure_to_lu(self, pressure_pu):
        return pressure_pu / self.characteristic_pressure_pu * self.characteristic_pressure_lu

    def convert_length_to_pu(self, length_lu):
        return length_lu * self.characteristic_length_pu / self.characteristic_length_lu

    def convert_length_to_lu(self, length_pu):
        return length_pu * self.characteristic_length_lu / self.characteristic_length_pu

    def convert_energy_to_pu(self, energy_lu):
        """Energy is defined here in units of [density * velocity**2]"""
        return (
            energy_lu * (self.characteristic_density_pu * self.characteristic_velocity_pu**2)
            / (self.characteristic_density_lu * self.characteristic_velocity_lu ** 2)
        )

    def convert_energy_to_lu(self, energy_pu):
        """Energy is defined here in units of [density * velocity**2]"""
        return (
            energy_pu * (self.characteristic_density_lu * self.characteristic_velocity_lu**2)
            / (self.characteristic_density_pu * self.characteristic_velocity_pu ** 2)
        )

    def convert_incompressible_energy_to_pu(self, energy_lu):
        """Energy in incompressible systems is defined in units of [velocity**2]"""
        return energy_lu * (self.characteristic_velocity_pu**2) / (self.characteristic_velocity_lu ** 2)

    def convert_incompressible_energy_to_lu(self, energy_pu):
        """Energy in incompressible systems is defined in units of [velocity**2]"""
        return energy_pu * (self.characteristic_velocity_lu**2) / (self.characteristic_velocity_pu ** 2)
