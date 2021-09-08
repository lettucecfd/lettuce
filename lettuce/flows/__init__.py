"""
Example flows.
"""

from .taylorgreen import TaylorGreenVortex2D, TaylorGreenVortex3D
from .couette import CouetteFlow2D
from .obstacle import Obstacle2D, Obstacle3D
from .poiseuille import PoiseuilleFlow2D
from .doublyshear import DoublyPeriodicShear2D
from .decayingturbulence import DecayingTurbulence

from .util import flow_by_name
