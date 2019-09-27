
from lettuce.equilibrium import QuadraticEquilibrium
import torch

class Guo:
    def __init__(self, lattice, tau, F):
        self.lattice = lattice
        self.tau = tau
        self.F = F
        self.A = 0.5

    def __call__(self, f, u):
        #u = self.lattice.u_force(f, self)

        exu = self.lattice.einsum("ia,a->i", [self.lattice.e,u])
        euxc= self.lattice.einsum("i,ia->a", [exu,self.lattice.e])

        uu= self.lattice.einsum("ia,a->ia", [torch.ones([self.lattice.Q,self.lattice.D]),u])
        ee= self.lattice.einsum("ia,ia->ia", [torch.ones([self.lattice.Q, self.lattice.D, f.shape[1], f.shape[2]]), self.lattice.e]) #Dritte Dimension berücksichtigen
        emu= ee - uu

        temp = self.lattice.einsum("i,ia->ia", [self.lattice.w ,emu / self.lattice.cs**2 + euxc / self.lattice.cs**4])
        temp_2 = self.lattice.einsum("ia,a->i", [temp, self.F])

        return (1-1/2*self.tau)*temp_2