
__all__ = ["DomainDecomposition", "MPIObservableReporter", "MPIStreaming"]

from .domain import Domain, BoxDomain
from typing import Sequence, Optional
import os
import sys
import torch.distributed as dist
import torch
import copy
import numpy as np

#
# TODO: (v0.1.5) MPI Streaming
# TODO: (v0.1.5) Upper class for boundaries
# TODO: (v0.1.5) Reporter for observables
# TODO: (v0.1.5) Visualization - VTK Files

# for student
# TODO: (v0.2.0) Expand attribute dims for 2nd, 3rd dimension (splitting)
# TODO: (v0.2.0) Create map, coord, topology
# TODO: (v0.2.0) Create Fct. for flow-decomposition
# TODO: (v0.2.0) Syntax for Fct names
# TODO: (v0.2.0) Add utilities from mpi4py)

class DomainDecomposition:
    def __init__(self,
                 domain: "Domain" = None,
                 flow: "Flow" = None,
                 dims: Sequence[int] = None,
                 mpi: bool = False,
                 src: int = 0,):
        self.flow = flow
        # self.domain = domain if domain else flow.domain
        self.domain = domain
        self.domains = []
        self._dims = dims
        self.mpi = mpi
        self.src = src
        self._mpi_rank = domain.rank
        if self.mpi:
            self._mpi_rank = dist.get_rank()
            self._mpi_size = dist.get_world_size()
            self._mpi_group = dist.group.WORLD
            self.init_mpi_process()
            self._dims[0] = self._mpi_size #TODO: only for dim==0
            self._placeholder = []
        else:
            self._mpi_rank = domain.rank

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, new_dim):
        self._dims = new_dim

    @property
    def topo(self):
        return self._topo

    @topo.setter
    def topo(self, new_topo):
        self._topo = new_topo

    @property
    def mpi_size(self):
        if self._mpi_size is None:
            raise Exception("MPI distribution is not initialized.")
        else:
            return self._mpi_size

    @mpi_size.setter
    def mpi_size(self, new_size: int):
        self._mpi_size = new_size

    @property
    def mpi_rank(self) -> int:
        if self._mpi_rank is None:
            raise Exception("MPI distribution is not initialized.")
        else:
            return self._mpi_rank

    @mpi_rank.setter
    def mpi_rank(self, new_rank: int):
        if new_rank >= self.mpi_size:
             raise ValueError(f"mpi_rank ({new_rank}) has to be smaller than mpi_size ({self.mpi_size()})")
        self._rank = new_rank

    @property
    def mpi_group(self):
        return self._mpi_group

    @mpi_group.setter
    def mpi_group(self, new_group):
        self._mpi_group = new_group

    def init_mpi_process(self,
                         rank=-1,
                         size=-1,
                         group=None,
                         backend="mpi"):
        # pass
        """ Initialize the distributed environment. """
        # os.environ['MASTER_ADDR'] = '127.0.0.1'
        # os.environ['MASTER_PORT'] = '29500'
        # dist.init_process_group(backend=backend,
        #                         rank=rank,
        #                         world_size=size)
        self.mpi_size = dist.get_world_size()
        self.mpi_rank = dist.get_rank()
        self.mpi_group = dist.group.WORLD if group is None else group

    def split_domain(self, *coordinates, split_flow: bool = False) -> Sequence["BoxDomain"]:
        """Split domain
        Notes:
            - works currently only for dim==0
            """
        if self.dims[0] == 1:
            raise ValueError(f"Dims: {self.dims} is to low.")
        if self.mpi_rank == 0:
            if len(coordinates)==0:
                coordinates = ((self.domain.grid()[0][np.linspace(0, self.domain.resolution[0], self.dims[0] + 1, dtype=int), 0])[1:-1] if self.domain.dim==2 else
                               (self.domain.grid()[0][np.linspace(0, self.domain.resolution[0], self.dims[0] + 1, dtype=int), 0, 0])[1:-1]
                )
            domains = self.domain.split(*coordinates)
            self._placeholder = [torch.tensor(_.shape,dtype=int) for _ in domains]
            for i in range(len(self._placeholder)):
                self._placeholder[i][0] -= 1
            self.create_map(domains)
            flows = self.split_flow(domains) if split_flow else None
        else:
            domains = None
            flows = None
        if self.mpi:
            domain = self.send(domains)
            if split_flow:
                flow = self.send(flows)
            self.create_map(domain, mpi=True)
            return (domain, flow) if split_flow else domain
        else:
            return (domains, flows) if split_flow else domains

    def split_flow(self, domains, flow = None) -> Sequence["BoxDomain"]:
        # TODO: This might be a memory problem
        _flow = self.flow if flow is None else flow
        if _flow is None:
            raise ValueError("There is no flow to split")
        flows = []
        counter = 0
        for domain_id, domain in enumerate(domains):
            flow = copy.deepcopy(_flow)
            for i in range(self.domain.shape[0] - domain.shape[0] + 1):
                fit = torch.all(torch.isclose(self.domain.grid()[0][i:i + domain.shape[0], ...], domain.grid()[0]))
                if fit:
                    counter += 1
                    flow.f = _flow.f[:, i:i + domain.shape[0], ...].clone()
                    flow.domain = copy.deepcopy(domain)
                    flow.grid = self.domain.grid(as_numpy=True)[0][i:i + domain.shape[0], ...]
            flows.append(flow)
        if domain_id + 1 is not counter:
            raise ValueError(
                f"Flow is not splitted correctly. "
                f"Domain is splitted into {len(domains)} domains. "
                f"Flow is splitted into {counter} flows."
            )
        return flows

    def send(self, objects):
        """Sending domains to devices"""
        if self.mpi_rank == 0:
            print("Sending",objects[0].__class__.__name__,"to devices.")
        else:
            objects = [None] * self.mpi_size
        object = [None]
        dist.scatter_object_list(object, objects, src=self.src, group=self.mpi_group)
        object[0].rank = self.mpi_rank
        return object

    def create_map(self, domains, mpi = False):
        """Create a map with coordination"""
        #TODO: Naiv hard coded implementiert. Hier muss überprüft werden, auf welche Ranks die Domains verteilt wurden.
        self.topo = torch.zeros(self.dims)
        if mpi:
            self.topo = torch.arange(torch.prod(torch.tensor(self.dims))).view(self.dims)
            self.left_neighbor = torch.roll(self.topo,1,0)
            self.right_neighbor = torch.roll(self.topo, 1, 0)

    def get_topo(self):
        """Return information on the cartesian topology"""
        # TODO: Decom - Implement fct. - get_topo
        return self.topo

    def get_neighbors(self):
        """Return information on the cartesian topology"""
        left_neighbor = torch.roll(self.topo, 1, 0)
        right_neighbor = torch.roll(self.topo, -1, 0)
        return left_neighbor, right_neighbor

    def get_coords(self, rank: int):
        """Translate ranks to logical coordinates"""
        #TODO: Decom - Implement fct. - get_coords
        pass

class MPIStreaming:
    def __init__(self, lattice, decom, device):
        self.lattice = lattice
        self._no_stream_mask = None
        self.decom = decom
        self.device = device+":"+str(decom.mpi_rank)
        # self.device = "cpu"

    @property
    def no_stream_mask(self):
        return self._no_stream_mask

    @no_stream_mask.setter
    def no_stream_mask(self, mask):
        self._no_stream_mask = mask

    def __call__(self, f):
        for i in range(1, self.lattice.Q):
            if self.no_stream_mask is None:
                f[i] = self._stream(f, i)
            else:
                new_fi = self._stream(f, i)
                f[i] = torch.where(self.no_stream_mask[i], f[i], new_fi)
        f = self._communication(f).clone()
        return f

    def _stream(self, f, i):
        return torch.roll(f[i], shifts=tuple(self.lattice.stencil.e[i]), dims=tuple(np.arange(self.lattice.D)))

    def _communication(self, f):
        left_neighbor, right_neighbor = self.decom.get_neighbors()

        # print(f"I'm {self.decom.mpi_rank} and my left_neighbor is {left_neighbor[self.decom.mpi_rank]}")
        # print(f"I'm {self.decom.mpi_rank} and my right is {right_neighbor[self.decom.mpi_rank]}")
        f_to_right = f[self.lattice.e[:, 0] == 1, -1, ...].cpu().clone().detach()
        f_from_left = torch.zeros_like(f_to_right).clone().detach()
        # f_from_left = torch.zeros_like(left_neighbor[self.decom.mpi_rank])
        to_right = dist.isend(tensor=f_to_right, dst=right_neighbor[self.decom.mpi_rank])
        # to_right = dist.isend(tensor=torch.tensor(self.decom.mpi_rank), dst=right_neighbor[self.decom.mpi_rank])
        from_left = dist.irecv(tensor=f_from_left, src=left_neighbor[self.decom.mpi_rank])
        to_right.wait()
        from_left.wait()
        f[self.lattice.e[:, 0] == 1, 0, ...] = f_from_left.detach().to(device=self.device)
        #print(f"Im rank {self.decom.mpi_rank} and my info is: ", f_from_left)

        f_to_left = f[self.lattice.e[:, 0] == -1, 0, ...].cpu().clone().detach()
        f_from_right = torch.zeros_like(f_to_left).clone().detach()
        to_left = dist.isend(tensor=f_to_left, dst=left_neighbor[self.decom.mpi_rank])
        from_right = dist.irecv(tensor=f_from_right, src=right_neighbor[self.decom.mpi_rank])
        to_left.wait()
        from_right.wait()
        f[self.lattice.e[:,0] == -1, -1, ...] = f_from_right.detach().to(device=self.device)


        return f

class MPIObservableReporter:
    def __init__(self, observable, decomposition, interval=1, out=sys.stdout):
        self.observable = observable
        self.decomposition = decomposition
        self.interval = interval
        self.out = [] if out is None else out
        self._parameter_name = observable.__class__.__name__
        #print('steps    ', 'time    ', self._parameter_name)

    def __call__(self, i, t, f):
        if i % self.interval == 0:
            f_send = f[:, :-1, ...].detach().clone().cpu()

            index = [f.shape[0]] + [-1] * self.decomposition.domain.dim
            f_all = [torch.zeros(tuple(_), dtype=f_send.dtype, device=f_send.device)[None, ...].expand(index).clone() for _ in
                     self.decomposition._placeholder] if self.decomposition.mpi_rank == 0 else None

            torch.distributed.gather_object(obj=f_send, object_gather_list=f_all, dst=0)
            if self.decomposition.mpi_rank == 0:
                ff = torch.cat(f_all, dim=1).to(device=f.device)
                # print("f_all0.shape ", f_all[0].shape)
                # print(torch.all(f_all[0] == f_send))
                # print("f_all1.shape ", f_all[1].shape)
                # print("shape:", ff.shape)
                # mass = self.observable.lattice.rho(ff)
                # print("mass", torch.sum(mass,dim=(0,1,2,3)))
                print(t)
                observed = self.observable.lattice.convert_to_numpy(self.observable(ff))
                assert len(observed.shape) < 2
                if len(observed.shape) == 0:
                    observed = [observed.item()]
                else:
                    observed = observed.tolist()
                entry = [i, t] + observed
                if isinstance(self.out, list):
                    self.out.append(entry)
                else:
                    print(*entry, file=self.out)