
from lettuce.equilibrium import QuadraticEquilibrium
import torch

class guo:
    def __init__(self, lattice, tau, F):
        self.lattice = lattice
        self.tau = tau
        self.F = F

    def __call__(self, f):

        import torch as tr
        #u = self.lattice.u(f)
        u = self.lattice.u_guo(f, self.F)

        exu = self.lattice.einsum("ia,a->i", [self.lattice.e,u])
        euxc= self.lattice.einsum("i,ia->a", [exu,self.lattice.e])

        uu= self.lattice.einsum("ia,a->ia", [torch.ones([self.lattice.Q,self.lattice.D]),u])
        ee= self.lattice.einsum("ia,ia->ia", [torch.ones([self.lattice.Q, self.lattice.D, f.shape[1], f.shape[2]]), self.lattice.e]) #Dritte Dimension berücksichtigen
        emu= ee - uu

        #F = tr.tensor([0, 0.0001])
        #test = euxc.repeat(9,1,1,1)
        temp = self.lattice.einsum("i,ia->ia", [self.lattice.w ,emu / self.lattice.cs**2 + euxc / self.lattice.cs**4])
        temp_2 = self.lattice.einsum("ia,a->i", [temp, self.F])

        f = 1
        #f = (1-0.5*1/self.tau)*self.lattice.einsum("q,q->q",self.lattice.w, (cmu) / self.lattice.cs ** 2 + cxuxc/ self.lattice.cs ** 4)
        return (1-1/2*self.tau)*temp_2