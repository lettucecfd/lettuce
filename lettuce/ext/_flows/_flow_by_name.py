from typing import Dict, Tuple, Type, AnyStr

from ... import Stencil
from .. import D2Q9, D3Q19
from . import (ExtFlow, TaylorGreenVortex, PoiseuilleFlow2D,
               DoublyPeriodicShear2D, CouetteFlow2D, DecayingTurbulence, LambOseenVortex2D)

__all__ = ['flow_by_name']

flow_by_name: Dict[AnyStr, Tuple[Type['ExtFlow'], Type['Stencil']]] = {
    'taylor2d': (TaylorGreenVortex, D2Q9),
    'taylor3d': (TaylorGreenVortex, D3Q19),
    'poiseuille2d': (PoiseuilleFlow2D, D2Q9),
    'shear2d': (DoublyPeriodicShear2D, D2Q9),
    'couette2d': (CouetteFlow2D, D2Q9),
    'decay2d': (DecayingTurbulence, D2Q9),
    'lamboseen': (LambOseenVortex2D, D2Q9),}
