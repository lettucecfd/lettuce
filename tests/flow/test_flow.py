from tests.common import *


@pytest.mark.parametrize("flowname", flow_by_name.keys())
def test_flow(flowname, fix_device, fix_dtype):
    IncompressibleFlow, stencil = flow_by_name[flowname]
    stencil = stencil() if callable(stencil) else stencil
    context = Context(device=fix_device, dtype=fix_dtype, use_native=False)
    flow = IncompressibleFlow(context=context, resolution=[16] * stencil.d,
                              reynolds_number=1, mach_number=0.05,
                              stencil=stencil)
    collision = BGKCollision(tau=flow.units.relaxation_parameter_lu)
    simulation = Simulation(flow=flow, collision=collision, reporter=[])
    simulation(1)

