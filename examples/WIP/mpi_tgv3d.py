import torch
import torch.distributed as dist
import torchdd as dd
import lettuce as lt
import numpy as np
import matplotlib.pyplot as plt

res = 50
time = 1 #sec
step = None
device = "cpu"
interval = 50
re = 400

dist.init_process_group(backend="mpi", rank=-1, world_size=-1)
dtype = torch.float64
domain = dd.BoxDomain(
    lower=torch.zeros(3),
    upper=2 * np.pi * torch.ones(3),
    resolution=torch.Size([res]*3),
    n_ghost=[[0, 1], [0, 0], [0, 0]],
    mpi_rank=0,
    device="cpu",
    dtype=torch.float64,
    endpoint=False)

lattice_cpu = lt.Lattice(lt.D3Q27, device="cpu", dtype=dtype)
lattice_gpu = lt.Lattice(lt.D3Q27, device=device+":"+str(dist.get_rank()), dtype=dtype)

flow = lt.TaylorGreenVortex3D(lattice=lattice_gpu,
                              domain=domain,
                              mach_number=0.05,
                              reynolds_number=re,
                              compute_f=False)

decom = dd.DomainDecomposition(domain=domain,
                               flow=flow,
                               dims=[1,1,1],
                               mpi=True)
domains = decom.split_domain()

flows = lt.TaylorGreenVortex3D(lattice=lattice_gpu,
                              domain=domains[0],
                              mach_number=0.05,
                              reynolds_number=re,
                              compute_f=False)

flows.units = flow.units

for i in domains:
    print("I'm ",flows.__class__.__name__," on rank:", i.rank," with ",flows.domain,", coordinates:", flows.domain.coord)

collision = lt.BGKCollision(lattice_gpu, tau=flows.units.relaxation_parameter_lu)
# streaming = lt.StandardStreaming(lattice_gpu)
streaming = dd.MPIStreaming(lattice=lattice_gpu, decom=decom, device=device)
simulation = lt.Simulation(flow=flows, lattice=lattice_gpu,  collision=collision, streaming=streaming)

energy = lt.IncompressibleKineticEnergy(lattice_gpu, flow)
reporter = dd.MPIObservableReporter(energy, decomposition=decom, interval=interval,)
simulation.reporters.append(reporter)

steps = int(flow.units.convert_time_to_lu(time)) if step is None else step
if dist.get_rank() == 0:
    print("steps:",steps)
print(f"Start simulation on rank: {domains[0].rank}")
mlups = simulation.step(steps)
print(f"Finish with {mlups} MLUPS")
