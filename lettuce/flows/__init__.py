"""
Example flows.
"""

from lettuce.flows.taylorgreen import TaylorGreenVortex2D, TaylorGreenVortex3D
from lettuce.flows.couette import CouetteFlow2D
from lettuce.flows.channel import ChannelFlow2D

def flow_by_name(flow_class=None):
    FlowDictorary = {
        "TGV2D": TaylorGreenVortex2D,
        "TCF2D": ChannelFlow2D,
        "COU2D": CouetteFlow2D}
    return FlowDictorary if flow_class is None else FlowDictorary[flow_class]