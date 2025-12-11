

import torch
import numpy as np
import os

from typing import List

from lettuce._simulation import Reporter, Simulation

class ProfileReporter(Reporter):
    
    def __init__(self, interval: int, flow, position_lu, i_start):
        super().__init__(interval)
        self.flow = flow
        self.i_start = i_start
        self.x_position = int(round(position_lu, 0))

        # linear interpolation of u if x_pos is off grid
        # ... positions x_pos# and weights w#
        self.interpol = False
        if position_lu % 1 != 0:
            self.interpol = True
            self.x_pos1 = int(np.floor(position_lu))
            self.x_pos2 = int(np.ceil(position_lu))
            self.w2 = position_lu - self.x_pos1
            self.w1 = 1 - self.w2
            print("ProfileReporter: requested position is off grid! "
                  "Interpolating u(" + str(position_lu) + ") = " 
                  + str(self.w1) + " * u(" + str(self.x_pos1) + ") + " 
                  + str(self.w2) + " * u(" + str(self.x_pos2) + ")")
        
        self.i_out = []
        self.out = []

    def __call__(self, simulation: 'Simulation'):
        if simulation.flow.i % self.interval == 0 and simulation.flow.i >= self.i_start:
            # calculate or interpolate velocity profile in y-direction at position
            if self.interpol:
                u = self.flow.u(self.flow.f)[:, self.x_pos1] * self.w1 + \
                    self.flow.u(self.flow.f)[:, self.x_pos2] * self.w2
            else:
                u = self.flow.u(self.flow.f)[:, self.x_position]
            u = self.flow.units.convert_velocity_to_pu(u).cpu().numpy()
            self.i_out.append(self.flow.i)
            if self.flow.stencil.d == 2:
                self.out.append(u)
            elif self.flow.stencil.d == 3:
                # average over z-axis (axial cross stream for cylinder obstacle)
                self.out.append(np.mean(u, axis=2))

