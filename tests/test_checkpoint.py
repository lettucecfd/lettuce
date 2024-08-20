from copy import deepcopy

from tests.common import *

def test_checkpoint(tmpdir):
    context = Context()
    flow = PoiseuilleFlow2D(context=context,
                            resolution=[512, 512],
                            reynolds_number=1,
                            mach_number=0.02,
                            initialize_with_zeros=False)
    f0 = deepcopy(flow.f)
    filename = tmpdir / 'PoiseuilleFlow2D'
    flow.dump(filename)
    simulation = Simulation(flow,
                            BGKCollision(flow.units.relaxation_parameter_lu),
                            [])
    simulation(10)
    flow.load(filename)
    assert torch.eq(f0, flow.f).all()

    """
    This could be a way to test a dump-load workflow which stores all flow 
    attributes. I did not get it to work, yet.
    context = Context('cuda')
    flow = PoiseuilleFlow2D(context=context,
                            resolution=16,
                            reynolds_number=1,
                            mach_number=0.02,
                            initialize_with_zeros=False)
    f0 = deepcopy(flow.f)
    filename = './PoiseuilleFlow2D'
    flow.dump(filename)

    simulation = Simulation(flow,
                            BGKCollision(flow.units.relaxation_parameter_lu),
                            [])
    simulation(10)

    context2 = Context('cpu')
    flow2 = PoiseuilleFlow2D(context=context2,
                             resolution=32,
                             reynolds_number=10,
                             mach_number=0.01,
                             initialize_with_zeros=True)
    flow2.load(filename)

    assert flow2.resolution == flow.resolution
    assert flow2.units.reynolds_number == flow.units.reynolds_number
    assert flow2.units.mach_number == flow.units.mach_number
    assert flow2.initialize_with_zeros == flow.initialize_with_zeros
    assert torch.eq(f0, flow2.f).all()
    """