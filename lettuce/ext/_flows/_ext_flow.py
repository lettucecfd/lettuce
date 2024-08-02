from abc import ABC, abstractmethod
from typing import List, Optional, Union

from .. import D1Q3, D2Q9, D3Q19, QuadraticEquilibrium
from ... import Flow


class ExtFlow(Flow, ABC):
    """
    Most __init__ methods of Flow look the same while having a lot of parameters.
    This class implements a common parameter set and allows
    subclasses to only implement the creation of the unit conversion and resolution.
    """

    def __init__(self, context: 'Context', resolution: Union[int, List[int]], reynolds_number, mach_number, stencil: Optional['Stencil'] = None,
                 equilibrium: Optional['Equilibrium'] = None):
        # set _stencil or default _stencil based on dimension
        resolution = self.make_resolution(resolution, stencil)
        assert len(resolution) in [1, 2, 3], f"flow supports dimensions 1, 2 and 3 but {len(resolution)} dimensions where requested."
        default_stencils = [D1Q3(), D2Q9(), D3Q19()]
        stencil = stencil or default_stencils[len(resolution) - 1]
        stencil = stencil() if callable(stencil) else stencil

        # set _equilibrium or quadratic _equilibrium
        equilibrium = equilibrium or QuadraticEquilibrium()
        Flow.__init__(self, context, resolution, self.make_units(reynolds_number, mach_number, resolution), stencil, equilibrium)

    @abstractmethod
    def make_resolution(self, resolution: Union[int, List[int]], stencil: Optional['Stencil'] = None) -> List[int]:
        ...

    @abstractmethod
    def make_units(self, reynolds_number, mach_number, resolution: List[int]) -> 'UnitConversion':
        ...
