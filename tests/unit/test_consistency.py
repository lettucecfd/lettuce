from tests.common import *
from tests.unit import create_default_unit_conversion


def test_consistency():
    rho = 0.9
    u = 0.1
    units = create_default_unit_conversion()

    assert (units.convert_density_to_pu(rho)
            * units.convert_velocity_to_pu(u) ** 2
            == pytest.approx(units.convert_energy_to_pu(rho * u * u))
            )
    assert (units.convert_velocity_to_pu(u) ** 2
            == pytest.approx(units.convert_incompressible_energy_to_pu(u * u)))
