from tests.common import *


def test_getitem(fix_device, fix_dtype):
    moments = D2Q9Lallemand(D2Q9(), Context(fix_device, fix_dtype))
    assert moments["jx", "jy"] == [1, 2]
    assert moments["rho"] == [0]
