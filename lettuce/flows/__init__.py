"""
Example flows.
"""

from lettuce.flows.taylorgreen import TaylorGreenVortex2D, TaylorGreenVortex3D
from lettuce.flows.couette import CouetteFlow2D
from lettuce.flows.poiseuille import PoiseuilleFlow2D
from lettuce.flows.doublyshear import DoublyPeriodicShear2D
from lettuce.flows.decayingturbulence import DecayingTurbulence
from lettuce.flows.obstacle import Obstacle, Obstacle2D, Obstacle3D
from lettuce.stencils import D2Q9, D3Q19

flow_by_name = {
    "taylor2D": [TaylorGreenVortex2D, D2Q9],
    "taylor3D": [TaylorGreenVortex3D, D3Q19],
    "poiseuille2D": [PoiseuilleFlow2D, D2Q9],
    "shear2D": [DoublyPeriodicShear2D, D2Q9],
    "couette2D": [CouetteFlow2D, D2Q9],
    "decay": [DecayingTurbulence, D2Q9]
}
