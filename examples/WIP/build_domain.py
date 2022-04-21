import torchdd as dd
import torch
import torch.distributed as dist
import lettuce as lt
import numpy as np

device = "cpu"
dtype = torch.float64

lattice = lt.Lattice(lt.D3Q27, device, dtype)
domain = dd.BoxDomain(
            lower=torch.zeros(2),
            upper=2*torch.ones(2),
            resolution=torch.Size([4, 4]),
            n_ghost=[[0, 1], [0, 0]],
            mpi_rank=0,
            device=device,
            dtype=torch.float64,
            endpoint=False)

print(domain.grid(with_ghost=True))
print("domain device:", domain.device)
print("domain coord:", domain.coord)
print("domain dim:", domain.dim)
print("domain lower:", domain.lower)
print("domain upper:", domain.upper)
print("domain resolution:", domain.resolution)
print("domain shape:", domain.shape)
print("domain tensor_shape:", domain.tensor_shape)
print("domain lengths:", domain.lengths)
print("domain cell_lengts:", domain.cell_lengths)
print("domain n_cells:", domain.n_cells)
print("domain n_points:", domain.n_points)
print("domain n_ghosts:", domain.n_ghost)
