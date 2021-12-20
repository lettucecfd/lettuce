from . import *
from .. import D2Q9, D3Q19

flow_by_name = {
    "taylor2D": [TaylorGreenVortex2D, D2Q9],
    "taylor3D": [TaylorGreenVortex3D, D3Q19],
    "poiseuille2D": [PoiseuilleFlow2D, D2Q9],
    "shear2D": [DoublyPeriodicShear2D, D2Q9],
    "couette2D": [CouetteFlow2D, D2Q9],
    "decay": [DecayingTurbulence, D2Q9]
}
