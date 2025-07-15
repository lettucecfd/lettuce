from typing import Dict, Tuple, Type

from ... import Stencil
from .. import D2Q9, D3Q19, D3Q27
from . import (ExtFlow, TaylorGreenVortex, PoiseuilleFlow2D,
               DoublyPeriodicShear2D, CouetteFlow2D, DecayingTurbulence)

__all__ = ['flow_by_name']

flow_by_name: Dict[str, Tuple[Type['ExtFlow'], Type['Stencil']]] = {
    'taylor2d': (TaylorGreenVortex, D2Q9),
    'taylor3d_d3q19': (TaylorGreenVortex, D3Q19),
    'taylor3d_d3q27': (TaylorGreenVortex, D3Q27),
    'poiseuille2d': (PoiseuilleFlow2D, D2Q9),
    'shear2d': (DoublyPeriodicShear2D, D2Q9),
    'couette2d': (CouetteFlow2D, D2Q9),
    'decay2d': (DecayingTurbulence, D2Q9)}

