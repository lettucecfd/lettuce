"""
Test unit conversion.
"""

from lettuce import UnitConversion

import pytest

import numpy as np


def test_reynolds_number_consistent(lattice):
    units = UnitConversion(lattice, reynolds_number=1000, mach_number=0.05, characteristic_length_lu=100,
                           characteristic_length_pu=2*np.pi, characteristic_velocity_pu=2)
    re_lu = units.characteristic_velocity_lu * units.characteristic_length_lu / units.viscosity_lu
    re_pu = units.characteristic_velocity_pu * units.characteristic_length_pu / units.viscosity_pu
    assert re_lu == pytest.approx(re_pu)


def test_conversion_reversible(lattice):
    units = UnitConversion(lattice, reynolds_number=1000, mach_number=0.05, characteristic_length_lu=100,
                           characteristic_length_pu=2*np.pi, characteristic_velocity_pu=2, characteristic_density_pu=0.7)
    assert units.convert_velocity_to_lu(units.convert_velocity_to_pu(2.0)) == pytest.approx(2.0)
    assert units.convert_time_to_lu(units.convert_time_to_pu(2.0)) == pytest.approx(2.0)
    assert units.convert_coordinates_to_lu(units.convert_coordinates_to_pu(2.0)) == pytest.approx(2.0)
    assert units.convert_density_to_lu(units.convert_density_to_pu(2.0)) == pytest.approx(2.0)
    assert units.convert_pressure_to_lu(units.convert_pressure_to_pu(2.0)) == pytest.approx(2.0)
    assert units.convert_density_lu_to_pressure_pu(units.convert_pressure_pu_to_density_lu(2.0)) == pytest.approx(2.0)
    assert units.convert_energy_to_lu(units.convert_energy_to_pu(2.0)) == pytest.approx(2.0)
    assert units.convert_incompressible_energy_to_lu(units.convert_incompressible_energy_to_pu(2.0)) == pytest.approx(2.0)


def test_consistency(lattice):
    units = UnitConversion(lattice, reynolds_number=1000, mach_number=0.05, characteristic_length_lu=100,
                           characteristic_length_pu=2*np.pi, characteristic_velocity_pu=2, characteristic_density_pu=0.7)
    rho = 0.9
    u = 0.1
    assert (
        units.convert_density_to_pu(rho)*units.convert_velocity_to_pu(u)**2
        == pytest.approx(units.convert_energy_to_pu(rho*u*u))
    )
    assert units.convert_velocity_to_pu(u)**2 == pytest.approx(units.convert_incompressible_energy_to_pu(u*u))
