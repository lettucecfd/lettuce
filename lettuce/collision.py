"""
Collision models
"""

from lettuce.equilibrium import IncompressibleQuadraticEquilibrium


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


class MRTCollision:
    def __init__(self, lattice, moment_transform, relaxation_parameters, moment_equilibrium=None):
        raise NotImplementedError


class BGKInitialization:
    def __init__(self, lattice, flow, moment_transformation):
        self.lattice = lattice
        self.tau = flow.units.relaxation_parameter_lu
        self.moment_transformation = moment_transformation
        p, u = flow.initial_solution(flow.grid)
        self.u = flow.units.convert_velocity_to_lu(lattice.convert_to_tensor(u))
        self.rho0 = flow.units.characteristic_density_lu
        self.equilibrium = IncompressibleQuadraticEquilibrium(self.lattice, rho0=self.rho0)

    def __call__(self, f):
        drho = self.lattice.rho(f)
        feq = self.equilibrium(drho, self.u)
        m = self.moment_transformation.transform(f)
        meq = self.moment_transformation.transform(feq)
        mnew = m - 1.0/self.tau * (m-meq)
        #f = f - 1.0/self.tau * (f-feq)
        mnew[0] = m[0] - 1.0/(self.tau+1) * (m[0]-meq[0])
        mnew[1:self.lattice.D+1] = drho*self.u # + self.u
        f = self.moment_transformation.inverse_transform(mnew)
        return f
