import sys
import torch

from ... import Reporter

__all__ = ['ErrorReporter']


class ErrorReporter(Reporter):
    """Reports numerical errors with respect to analytic solution."""

    def __init__(self, analytical_solution, interval=1, out=sys.stdout):
        Reporter.__init__(self, interval)
        self.out = [] if out is None else out
        if not isinstance(self.out, list):
            print("#error_u         error_p", file=self.out)

    def __call__(self, simulation: 'Simulation'):
        i, t, f = simulation.flow.i, simulation.units.convert_time_to_pu(
            simulation.flow.i), simulation.flow.f

        if i % self.interval == 0:
            pref, uref = simulation.flow.analytic_solution(t=t)
            pref = simulation.flow.context.convert_to_tensor(pref)
            uref = simulation.flow.context.convert_to_tensor(uref)
            u = simulation.flow.u_pu
            p = simulation.flow.p_pu

            resolution = torch.pow(torch.prod(
                simulation.flow.context.convert_to_tensor(p.size())),
                1 / simulation.flow.stencil.d)

            err_u = (torch.norm(u - uref)
                     / resolution ** (simulation.flow.stencil.d / 2))
            err_p = (torch.norm(p - pref)
                     / resolution ** (simulation.flow.stencil.d / 2))

            if isinstance(self.out, list):
                self.out.append([err_u.item(), err_p.item()])
            else:
                print(err_u.item(), err_p.item(), file=self.out)
