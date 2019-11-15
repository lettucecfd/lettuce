"""
Example flows.
"""

from lettuce.flows.taylorgreen import TaylorGreenVortex2D, TaylorGreenVortex3D
from lettuce.flows.couette import CouetteFlow2D
from lettuce.flows.poiseuille import PoiseuilleFlow2D


def flow_by_name(flow_class=None):
    flow_dictionary = {
        "taylor2D": TaylorGreenVortex2D,
        "poiseuille2D": PoiseuilleFlow2D,
        "couette2D": CouetteFlow2D}
    return flow_dictionary if flow_class is None else flow_dictionary[flow_class]
