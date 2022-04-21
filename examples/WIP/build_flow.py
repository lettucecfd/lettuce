import torch
import torchdd as dd
import lettuce as lt
import numpy as np

dtype = torch.float64
lattice_cpu = lt.Lattice(lt.D3Q27, device="cpu", dtype=dtype)

domain = dd.BoxDomain(
    lower=torch.zeros(3),
    upper=2 *np.pi* torch.ones(3),
    resolution=torch.Size([50, 50, 50]),
    endpoint=False,
    mpi_rank=0,
    device="cpu",
    dtype=torch.float64)

flow = lt.TaylorGreenVortex3D(lattice=lattice_cpu,
                              domain=domain,
                              mach_number=0.05,
                              reynolds_number=400,
                              compute_f=True)

print(flow.f.device)
print(flow.f.shape)