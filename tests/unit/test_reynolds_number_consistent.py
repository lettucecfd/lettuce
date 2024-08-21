from tests.common import *
from tests.unit import create_default_unit_conversion


def test_reynolds_number_consistent():
    units = create_default_unit_conversion()
    re_lu = units.characteristic_velocity_lu * units.characteristic_length_lu / units.viscosity_lu
    re_pu = units.characteristic_velocity_pu * units.characteristic_length_pu / units.viscosity_pu
    assert re_lu == pytest.approx(re_pu)
