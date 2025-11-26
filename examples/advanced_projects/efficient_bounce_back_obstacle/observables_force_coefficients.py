from typing import Optional

import torch
import numpy as np

from examples.advanced_projects.efficient_bounce_back_obstacle.obstacle_cylinder import ObstacleCylinder
from lettuce import Reporter, Flow, Observable

__all__ = ['DragCoefficient', 'LiftCoefficient']

#TODO: write this as "force"-coefficient and make X, Y, Z asl 0,1,2 direction choosable (parameter)
class DragCoefficient(Observable):
    """The drag coefficient of an obstacle, calculated using momentum exchange method (MEM, MEA)

    calculates the density, gets the force in x direction on the obstacle boundary,
    calculates the coefficient of drag
    """

    def __init__(self, flow, obstacle_boundary, solid_mask, area_pu):
        super().__init__(flow)
        self.obstacle_boundary = obstacle_boundary
        self.solid_mask = solid_mask
        self.area_lu = area_pu * (self.flow.units.characteristic_length_lu/self.flow.units.characteristic_length_pu) ** (self.flow.stencil.d-1) # cross-sectional area of obstacle in LU (! lengthdimension in 2D -> area-dimension = self.lattice.D-1)
        self.nan_tensor = self.context.convert_to_tensor(torch.nan)
        self.solid_mask = self.context.convert_to_tensor(self.solid_mask)

    def __call__(self, f: torch.Tensor = None):
        #OLD rho = torch.mean(self.lattice.rho(f[:, 0, ...]))  # simple rho_mean, including the boundary region
        # rho_mean (excluding (solid) boundary region, where values are non-physical):
        rho_tmp = torch.where(self.solid_mask, self.nan_tensor, self.flow.rho(f))
        rho_mean = torch.nanmean(rho_tmp)
        force_x_lu = self.obstacle_boundary.force_sum[0]  # get current force on obstacle in x direction
        drag_coefficient = force_x_lu / (0.5 * rho_mean * self.flow.units.characteristic_velocity_lu ** 2 * self.area_lu)  # calculate drag_coefficient in LU
        return drag_coefficient


class LiftCoefficient(Observable):
    """The drag coefficient of an obstacle, calculated using momentum exchange method (MEM, MEA)

    calculates the density, gets the force in x direction on the obstacle boundary,
    calculates the coefficient of drag
    """

    def __init__(self, flow, obstacle_boundary, solid_mask, area_pu):
        super().__init__(flow)
        self.obstacle_boundary = obstacle_boundary
        self.solid_mask = solid_mask
        self.area_lu = area_pu * (self.flow.units.characteristic_length_lu/self.flow.units.characteristic_length_pu) ** (self.flow.stencil.d-1) # cross-sectional area of obstacle in LU (! lengthdimension in 2D -> area-dimension = self.lattice.D-1)
        self.nan_tensor = self.context.convert_to_tensor(torch.nan)
        self.solid_mask = self.context.convert_to_tensor(self.solid_mask)

    def __call__(self, f: torch.Tensor = None):
        #OLD rho = torch.mean(self.lattice.rho(f[:, 0, ...]))  # simple rho_mean, including the boundary region
        # rho_mean (excluding (solid) boundary region, where values are non-physical):
        rho_tmp = torch.where(self.solid_mask, self.nan_tensor, self.flow.rho(f))
        rho_mean = torch.nanmean(rho_tmp)
        force_y_lu = self.obstacle_boundary.force_sum[1]  # get current force on obstacle in x direction
        lift_coefficient = force_y_lu / (0.5 * rho_mean * self.flow.units.characteristic_velocity_lu ** 2 * self.area_lu)  # calculate drag_coefficient in LU
        return lift_coefficient