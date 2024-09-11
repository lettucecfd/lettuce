from lettuce._flow import initialize_f_neq
from tests.conftest import *


class EinsumFlow(TestFlow):
    def __init__(self, context, res):
        super().__init__(context, resolution=res,
                         reynolds_number=1, mach_number=0.01)


    def j(self, f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """momentum"""
        return self.einsum("qd,q->d",
                           [self.torch_stencil.e, self.f if f is None else f])

    def incompressible_energy(self, f: Optional[torch.Tensor] = None
                              ) -> torch.Tensor:
        """incompressible kinetic energy"""
        f = self.f if f is None else f
        return 0.5 * self.einsum("d,d->", [self.u(f), self.u(f)])

    def entropy(self, f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """entropy according to the H-theorem"""
        f = self.f if f is None else f
        f_log = -torch.log(self.einsum("q,q->q",
                                       [f, 1 / self.torch_stencil.w]))
        return self.einsum("q,q->", [f, f_log])

    def pseudo_entropy_global(self, f: Optional[torch.Tensor] = None
                              ) -> torch.Tensor:
        """pseudo_entropy derived by a Taylor expansion around the weights"""
        f = self.f if f is None else f
        f_w = self.einsum("q,q->q", [f, 1 / self.torch_stencil.w])
        return self.rho() - self.einsum("q,q->", [f, f_w])

    def pseudo_entropy_local(self, f: Optional[torch.Tensor] = None
                             ) -> torch.Tensor:
        """pseudo_entropy derived by a Taylor expansion around the local
        equilibrium"""
        f = self.f if f is None else f
        f_feq = f / self.equilibrium(self)
        return self.rho(f) - self.einsum("q,q->", [f, f_feq])

    def shear_tensor(self, f: Optional[torch.Tensor] = None) -> torch.Tensor:
        """computes the shear tensor of a given self.f in the sense
        Pi_{\alpha \beta} = f_i * e_{i \alpha} * e_{i \beta}"""
        shear = self.einsum("qa,qb->qab",
                            [self.torch_stencil.e, self.torch_stencil.e])
        shear = self.einsum("q,qab->ab", [self.f if f is None else f, shear])
        return shear

    def einsum(self, equation, fields, *args) -> torch.Tensor:
        """Einstein summation on local fields."""
        inputs, output = equation.split("->")
        inputs = inputs.split(",")
        for i, inp in enumerate(inputs):
            if len(inp) == len(fields[i].shape):
                pass
            elif len(inp) == len(fields[i].shape) - self.stencil.d:
                inputs[i] += "..."
                if not output.endswith("..."):
                    output += "..."
            else:
                assert False, "Bad dimension."
        equation = ",".join(inputs) + "->" + output
        return torch.einsum(equation, fields, *args)

def initialize_f_neq_floweinsum(flow: 'EinsumFlow'):
    """Initialize the distribution function values. The f^(1) contributions are
    approximated by finite differences. See Krüger et al. (2017).
    """
    rho = flow.rho()
    u = flow.u()

    grad_u0 = torch_gradient(u[0], dx=1, order=6)[None, ...]
    grad_u1 = torch_gradient(u[1], dx=1, order=6)[None, ...]
    S = torch.cat([grad_u0, grad_u1])

    if flow.stencil.d == 3:
        grad_u2 = torch_gradient(u[2], dx=1, order=6)[None, ...]
        S = torch.cat([S, grad_u2])

    Pi_1 = (1.0 * flow.units.relaxation_parameter_lu * rho * S
            / flow.torch_stencil.cs ** 2)
    Q = (torch.einsum('ia,ib->iab',
                      [flow.torch_stencil.e, flow.torch_stencil.e])
         - torch.eye(flow.stencil.d, device=flow.torch_stencil.e.device)
         * flow.stencil.cs ** 2)
    Pi_1_Q = flow.einsum('ab,iab->i', [Pi_1, Q])
    fneq = flow.einsum('i,i->i', [flow.torch_stencil.w, Pi_1_Q])

    feq = flow.equilibrium(flow, rho, u)

    return feq - fneq


def initialize_f_neq_torcheinsum(flow: 'Flow'):
    """Initialize the distribution function values. The f^(1) contributions are
    approximated by finite differences. See Krüger et al. (2017).
    """
    rho = flow.rho()
    u = flow.u()

    grad_u0 = torch_gradient(u[0], dx=1, order=6)[None, ...]
    grad_u1 = torch_gradient(u[1], dx=1, order=6)[None, ...]
    S = torch.cat([grad_u0, grad_u1])

    if flow.stencil.d == 3:
        grad_u2 = torch_gradient(u[2], dx=1, order=6)[None, ...]
        S = torch.cat([S, grad_u2])

    Pi_1 = (1.0 * flow.units.relaxation_parameter_lu * rho * S
            / flow.torch_stencil.cs ** 2)
    Q = (torch.einsum('ia,ib->iab',
                      [flow.torch_stencil.e, flow.torch_stencil.e])
         - torch.eye(flow.stencil.d, device=flow.torch_stencil.e.device)
         * flow.stencil.cs ** 2)
    Pi_1_Q = torch.einsum('ab...,iab->i...', [Pi_1, Q])
    fneq = torch.einsum('i,i...->i...', [flow.torch_stencil.w, Pi_1_Q])

    feq = flow.equilibrium(flow, rho, u)

    return feq - fneq


class RegularizedEinsum(RegularizedCollision):
    def __call__(self, flow: 'Flow'):
        if self.Q_matrix is None:
            self.tau = flow.units.relaxation_parameter_lu
            self.Q_matrix = torch.zeros([flow.stencil.q, flow.stencil.d,
                                         flow.stencil.d],
                                        device=flow.context.device,
                                        dtype=flow.context.dtype)

            for a in range(flow.stencil.q):
                for b in range(flow.stencil.d):
                    for c in range(flow.stencil.d):
                        self.Q_matrix[a, b, c] = (
                                flow.torch_stencil.e[a, b]
                                * flow.torch_stencil.e[a, c])
                        if b == c:
                            self.Q_matrix[a, b, c] -= (flow.torch_stencil.cs
                                                       ** 2)
        feq = flow.equilibrium(flow)
        pi_neq = flow.shear_tensor(flow.f - feq)
        cs4 = flow.stencil.cs ** 4

        pi_neq = flow.einsum("qab,ab->q", [self.Q_matrix, pi_neq])
        pi_neq = flow.einsum("q,q->q", [flow.torch_stencil.w, pi_neq])

        fi1 = pi_neq / (2 * cs4)
        f = feq + (1. - 1. / self.tau) * fi1

        return f

class GuoEinsum(Guo):
    def source_term(self, u):
        emu = append_axes(self.flow.torch_stencil.e,
                          self.flow.torch_stencil.d) - u
        eu = self.flow.einsum("ib,b->i", [self.flow.torch_stencil.e, u])
        eeu = self.flow.einsum("ia,i->ia", [self.flow.torch_stencil.e, eu])
        emu_eeu = (emu / (self.flow.torch_stencil.cs ** 2)
                   + eeu / (self.flow.torch_stencil.cs ** 4))
        emu_eeuF = self.flow.einsum("ia,a->i", [emu_eeu, self.acceleration])
        weemu_eeuF = (append_axes(self.flow.torch_stencil.w,
                                  self.flow.torch_stencil.d)
                      * emu_eeuF)
        return (1 - 1 / (2 * self.tau)) * weemu_eeuF


@pytest.mark.parametrize("fix_dim", [1, 2, 3])
def test_einsum(fix_dim):
    context = Context(dtype=torch.float64)

    flow = EinsumFlow(context, [16] * fix_dim)
    flow.f = torch.rand_like(flow.f)

    f0 = flow.j(flow.f)
    f1 = flow.einsum("qd,q->d", [flow.torch_stencil.e, flow.f])
    f2 = torch.einsum("qd,q...->d...", [flow.torch_stencil.e, flow.f])
    assert torch.allclose(f0, f1)
    assert torch.allclose(f1, f2)

    f0 = flow.incompressible_energy(flow.f)
    # also in QuadraticEquilibrium, IncompressibleQuadraticEquilibrium,
    # QuadraticEquilibriumLessMemory
    f1 = 0.5 * flow.einsum("d,d->", [flow.u(flow.f), flow.u(flow.f)])
    f2 = 0.5 * torch.einsum("d...,d...->...", [flow.u(flow.f), flow.u(flow.f)])
    assert torch.allclose(f0, f1)
    assert torch.allclose(f1, f2)

    f0 = flow.entropy()
    # also __call__ in EquilibriumBoundaryPU, EquilibriumOutletP, MRTCollision,
    # RegularizedCollision, QuadraticEquilibrium,
    # IncompressibleQuadraticEquilibrium, QuadraticEquilibriumLessMemory,
    # test_equilibrium_boundary_pu_algorithm
    f_log = -torch.log(flow.einsum("q,q->q", [flow.f,
                                              1 / flow.torch_stencil.w]))
    f1 = flow.einsum("q,q->", [flow.f, f_log])
    f_log = -torch.log(torch.einsum("q...,q...->q...", [flow.f,
                                                        1 / flow.torch_stencil.w]))
    f2 = torch.einsum("q...,q...->...", [flow.f, f_log])
    assert torch.allclose(f0, f1)
    assert torch.allclose(f1, f2)

    f0 = flow.pseudo_entropy_global()
    f_w = flow.einsum("q,q->q", [flow.f, 1 / flow.torch_stencil.w])
    f1 = flow.rho() - flow.einsum("q,q->", [flow.f, f_w])
    f_w = torch.einsum("q...,q...->q...", [flow.f, 1 / flow.torch_stencil.w])
    f2 = flow.rho() - torch.einsum("q...,q...->...", [flow.f, f_w])
    assert torch.allclose(f0, f1)
    assert torch.allclose(f1, f2)

    f0 = flow.pseudo_entropy_local(flow.f)
    f_feq = flow.f / flow.equilibrium(flow)
    f1 = flow.rho(flow.f) - flow.einsum("q,q->", [flow.f, f_feq])
    f2 = flow.rho(flow.f) - torch.einsum("q...,q...->...", [flow.f, f_feq])
    assert torch.allclose(f0, f1)
    assert torch.allclose(f1, f2)

    f0 = flow.shear_tensor(flow.f)
    shear0 = flow.einsum("qa,qb->qab",
                        [flow.torch_stencil.e, flow.torch_stencil.e])
    shear0 = flow.einsum("q,qab->ab", [flow.f, shear0])
    f1 = shear0
    shear1 = torch.einsum("qa,qb->qab",
                         [flow.torch_stencil.e, flow.torch_stencil.e])
    shear1 = torch.einsum("q...,qab->ab...", [flow.f, shear1])
    f2 = shear1
    assert torch.allclose(f0, f1)
    assert torch.allclose(f1, f2)

    # in SmagorinskyCollision
    f0 = flow.einsum('ab,ab->', [shear0, shear0])
    f1 = torch.einsum('ab...,ab...->...', [shear1, shear1])
    assert torch.allclose(f0, f1)

    if fix_dim in [2, 3]:
        f0 = initialize_f_neq_floweinsum(flow)
        f1 = initialize_f_neq_torcheinsum(flow)
        f2 = initialize_f_neq(flow)
        assert torch.allclose(f0, f1)
        assert torch.allclose(f1, f2)

    # flow.einsum("qab,ab->q", [self.Q_matrix, pi_neq])
    # torch.einsum("qab...,ab...->q...", [self.Q_matrix, pi_neq])
    coll0 = RegularizedCollision(0.6)
    f0 = coll0(flow)
    coll1 = RegularizedEinsum(0.6)
    f1 = coll1(flow)
    assert torch.allclose(f0, f1)

    # from IncompressibleQuadraticEquilibrium
    exu0 = flow.einsum("qd,d->q", [flow.torch_stencil.e, flow.u()])
    exu1 = torch.einsum("qd,d...->q...", [flow.torch_stencil.e, flow.u()])
    assert torch.allclose(exu0, exu1)

    # Guo force source term
    a = [1] + [0] * (fix_dim - 1)
    guo0 = GuoEinsum(flow, 0.6, a)
    source_term0 = guo0.source_term(flow.u())
    guo1 = Guo(flow, 0.6, a)
    source_term1 = guo1.source_term(flow.u())
    assert torch.allclose(source_term0, source_term1)
