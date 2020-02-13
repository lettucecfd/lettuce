import lettuce as lt
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio

device=torch.device("cuda:0")
lattice = lt.Lattice(lt.D2Q9,device=device,dtype=torch.float64)

building = imageio.imread('47.png')
building = building.astype(bool)

#building = np.interp(building, (building.min(), building.max()), (-1, +1))

#Here you can change the resolution and the Reynolds number.
flow=lt.Obstacle2D(750,300,4500,0.1,lattice,building.shape[0])
x = flow.grid
mask_np = np.zeros([flow.resolution_x,flow.resolution_y],dtype=bool)
relative_position_x = int(mask_np.shape[0]/5-building.shape[0]/2)
relative_position_y = int(mask_np.shape[1]/2-building.shape[1]/2)

mask_np[relative_position_x:relative_position_x+building.shape[0],relative_position_y:relative_position_y+building.shape[1]]=building
mask_np=mask_np.astype(bool)
mask=mask_np.tolist()
mask=torch.tensor(mask,dtype=torch.uint8)

flow.mask=mask
collision=lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
streaming=lt.StandardStreaming(lattice)

simulation=lt.Simulation(flow,lattice,collision,streaming)
#simulation.initialize(max_num_steps=10)
print("Viscosity in lattice units:", flow.units.viscosity_lu)
for i in range(400):
    print(simulation.step(1000))
    print(i*1000,flow.getMaxU(simulation.f,lattice))
    print(flow.getSheddingFrequency(simulation.u_sample))
    u0 = (lattice.convert_to_numpy(lattice.u(simulation.f)[0])).transpose([1, 0])
    plt.imshow(u0)
    plt.savefig('test{}.png'.format(i))
probe= simulation.u_sample
np.save('probe',probe)

