import sys
import torch

from ... import Reporter

__all__ = ['ErrorReporter']


class ErrorReporter(Reporter):
    """Reports numerical errors with respect to analytic solution."""

    def __init__(self, flow, interval=1, out=sys.stdout):
        Reporter.__init__(self, interval)
        assert hasattr(flow, "analytic_solution")
        self.out = [] if out is None else out
        if not isinstance(self.out, list):
            print("#error_u         error_p", file=self.out)

    def __call__(self, flow: 'Flow'):
        i, t, f = flow.i, flow.units.convert_time_to_pu(flow.i), flow.f

        if i % self.interval == 0:
            pref, uref = flow.analytic_solution(t=t)
            pref = flow.context.convert_to_tensor(pref)
            uref = flow.context.convert_to_tensor(uref)
            u = flow.u_pu
            p = flow.p_pu

            resolution = torch.pow(torch.prod(flow.context.convert_to_tensor(p.size())), 1 / flow.stencil.d)

            err_u = torch.norm(u - uref) / resolution ** (flow.stencil.d / 2)
            err_p = torch.norm(p - pref) / resolution ** (flow.stencil.d / 2)

            if isinstance(self.out, list):
                self.out.append([err_u.item(), err_p.item()])
            else:
                print(err_u.item(), err_p.item(), file=self.out)
