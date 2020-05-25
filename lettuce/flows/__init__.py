"""
Example flows.
"""

from lettuce.flows.taylorgreen import TaylorGreenVortex2D, TaylorGreenVortex3D
from lettuce.flows.couette import CouetteFlow2D
from lettuce.flows.poiseuille import PoiseuilleFlow2D
from lettuce.flows.doublyshear import DoublyPeriodicShear2D

flow_by_name = {
    "taylor2D": TaylorGreenVortex2D,
    "poiseuille2D": PoiseuilleFlow2D,
    "shear2D": DoublyPeriodicShear2D,
    "couette2D": CouetteFlow2D
}
