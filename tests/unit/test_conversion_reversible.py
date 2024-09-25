from tests.conftest import *
from tests.unit import create_default_unit_conversion


def test_conversion_reversible():
    approx_two = pytest.approx(2.0)
    units = create_default_unit_conversion()

    assert approx_two == units.convert_velocity_to_lu(units.convert_velocity_to_pu(2.0))
    assert approx_two == units.convert_time_to_lu(units.convert_time_to_pu(2.0))
    assert approx_two == units.convert_length_to_lu(units.convert_length_to_pu(2.0))
    assert approx_two == units.convert_density_to_lu(units.convert_density_to_pu(2.0))
    assert approx_two == units.convert_pressure_to_lu(units.convert_pressure_to_pu(2.0))
    assert approx_two == units.convert_density_lu_to_pressure_pu(units.convert_pressure_pu_to_density_lu(2.0))
    assert approx_two == units.convert_energy_to_lu(units.convert_energy_to_pu(2.0))
    assert approx_two == units.convert_incompressible_energy_to_lu(units.convert_incompressible_energy_to_pu(2.0))
