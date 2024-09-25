from tests.conftest import *


def create_default_unit_conversion():
    return UnitConversion(
        reynolds_number=1000,
        mach_number=0.05,
        characteristic_length_pu=2 * np.pi,
        characteristic_velocity_pu=2,
        characteristic_length_lu=100,
        characteristic_density_pu=0.7)
