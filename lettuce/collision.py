"""
Collision models
"""

from lettuce.equilibrium import QuadraticEquilibrium
import torch


class BGKCollision:
    def __init__(self, lattice, tau):
        self.lattice = lattice
        self.tau = tau
    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u(f)
        feq = self.lattice.equilibrium(rho, u)
        f = f - 1.0/self.tau * (f-feq)
        return f

class BGKCollision_guo:
    def __init__(self, lattice, tau, force=None):
        self.force = force
        self.lattice = lattice
        self.tau = tau
    def __call__(self, f):
        rho = self.lattice.rho(f)
        u = self.lattice.u_force(f, force=self.force)
        feq = self.lattice.equilibrium(rho, u)

        #Is there a shorthand syntax or a cleaner way to to this?
        if self.force is not None:
            Si = self.force(f, u)
        else:
            Si = 0

        return f - 1.0/self.tau * (f-feq) + Si


class MRTCollision:
    """Multiple relaxation time collision operator

    This is an MRT operator in the most general sense of the word.
    The transform does not have to be linear and can, e.g., be any moment or cumulant transform.
    """
    def __init__(self, lattice, transform, relaxation_parameters):
        self.lattice = lattice
        self.transform = transform
        self.relaxation_parameters = lattice.convert_to_tensor(relaxation_parameters)

    def __call__(self, f):
        m = self.transform.transform(f)
        meq = self.transform.equilibrium(m)
        m = m - self.lattice.einsum("q,q->q", [1/self.relaxation_parameters, m-meq])
        f = self.transform.inverse_transform(m)
        return f


class BGKInitialization:
    """Keep velocity constant."""
    def __init__(self, lattice, flow, moment_transformation):
        self.lattice = lattice
        self.tau = flow.units.relaxation_parameter_lu
        self.moment_transformation = moment_transformation
        p, u = flow.initial_solution(flow.grid)
        self.u = flow.units.convert_velocity_to_lu(lattice.convert_to_tensor(u))
        self.rho0 = flow.units.characteristic_density_lu
        self.equilibrium = QuadraticEquilibrium(self.lattice)
        momentum_names = tuple([f"j{x}" for x in "xyz"[:self.lattice.D]])
        self.momentum_indices = moment_transformation[momentum_names]

    def __call__(self, f):
        rho = self.lattice.rho(f)
        feq = self.equilibrium(rho, self.u)
        m = self.moment_transformation.transform(f)
        meq = self.moment_transformation.transform(feq)
        mnew = m - 1.0/self.tau * (m-meq)
        mnew[0] = m[0] - 1.0/(self.tau+1) * (m[0]-meq[0])
        mnew[self.momentum_indices] = rho*self.u
        f = self.moment_transformation.inverse_transform(mnew)
        return f
