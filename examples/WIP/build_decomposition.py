import torch
import torch.distributed as dist
import torchdd as dd
import lettuce as lt
import numpy as np

res = 4
dist.init_process_group(backend="mpi", rank=-1, world_size=-1)
dtype = torch.float64
domain = dd.BoxDomain(
    lower=torch.zeros(3),
    upper=2*np.pi*torch.ones(3),
    resolution=torch.Size([res, res, res]),
    n_ghost=[[0, 1], [0, 0], [0, 0]],
    mpi_rank=0,
    device="cpu",
    dtype=dtype,
    endpoint=False)

lattice_cpu = lt.Lattice(lt.D3Q27, device="cpu", dtype=dtype)
lattice_gpu = lt.Lattice(lt.D3Q27, device="cpu", dtype=dtype)
flow = lt.TaylorGreenVortex3D(lattice=lattice_cpu,
                              domain=domain,
                              mach_number=0.1,
                              reynolds_number=400,
                              compute_f=True)

decom = dd.DomainDecomposition(domain=domain,
                               flow=flow,
                               dims=[2, 1, 1],
                               mpi=False)

# print(domain.grid(with_ghost=True)[0])
# print(domain.grid(with_ghost=True)[1])
# print()
# domains, flows = decom.split_domain(split_flow=True)
domains = decom.split_domain()


flows_0 = lt.TaylorGreenVortex3D(lattice=lattice_gpu,
                              domain=domains[0],
                              mach_number=0.05,
                              reynolds_number=400,
                              compute_f=True)

flows_1 = lt.TaylorGreenVortex3D(lattice=lattice_gpu,
                              domain=domains[1],
                              mach_number=0.05,
                              reynolds_number=400,
                              compute_f=True)

print(flows_0.f.shape)
print(flows_1.f.shape)

ff = torch.cat((flows_0.f[:,:-1,...], flows_1.f[:,:-1,...]), dim=1)
print(ff.shape)

# collision = lt.BGKCollision(lattice_gpu, tau=flows[0].units.relaxation_parameter_lu)
# streaming = lt.StandardStreaming(lattice_gpu)
# simulation = lt.Simulation(flow=flows[0], lattice=lattice_gpu,  collision=collision, streaming=streaming)
# mlups = simulation.step(1000)
# print(f"Finish with {mlups} MLUPS")
