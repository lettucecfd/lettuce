from typing import Dict, Tuple, Type, AnyStr

from ... import Stencil
from .. import D2Q9, D3Q19
from . import (ExtFlow, TaylorGreenVortex, PoiseuilleFlow2D,
               DoublyPeriodicShear2D, CouetteFlow2D, DecayingTurbulence)

__all__ = ['flow_by_name']

flow_by_name: Dict[AnyStr, Tuple[Type['ExtFlow'], Type['Stencil']]] = {
    'taylor2D': (TaylorGreenVortex, D2Q9),
    'taylor3D': (TaylorGreenVortex, D3Q19),
    'poiseuille2D': (PoiseuilleFlow2D, D2Q9),
    'shear2D': (DoublyPeriodicShear2D, D2Q9),
    'couette2D': (CouetteFlow2D, D2Q9),
    'decay': (DecayingTurbulence, D2Q9)}
