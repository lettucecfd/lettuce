from lettuce.util.moments import D2Q9Dellar, D2Q9Lallemand, moment_tensor
from tests.conftest import *


@pytest.mark.parametrize("MomentSet", (D2Q9Dellar, D2Q9Lallemand))
def test_conserved_moments_d2q9(MomentSet):
    multiindices = np.array([
        [0, 0], [1, 0], [0, 1]
    ])
    m = moment_tensor(D2Q9().e, multiindices)
    assert m == pytest.approx(MomentSet.matrix[:3, :])
