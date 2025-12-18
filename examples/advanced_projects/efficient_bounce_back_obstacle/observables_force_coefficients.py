"""Observables: force coefficients calculated from the force on an obstacle
    - coefficient of drag
    - coefficient of lift
"""

import torch
from lettuce import Flow, Observable

__all__ = ['DragCoefficient', 'LiftCoefficient']

# TODO (optional): write this as "force"-coefficient and make
#  X, Y, Z as 0,1,2 direction choosable (parameter) to unify drag and lift
class DragCoefficient(Observable):
    """The coefficient of drag of an obstacle, calculated using momentum exchange method (MEM, MEA)
        - calculates the density,
        - gets the force in x direction on the obstacle boundary,
        - calculates the coefficient of drag
    """

    def __init__(self, flow, obstacle_boundary, solid_mask, area_pu):
        super().__init__(flow)
        self.obstacle_boundary = obstacle_boundary
        self.solid_mask = solid_mask

        # cross-sectional area of obstacle in LU
        # (! mind length-dimension in 2D -> area-dimension = self.lattice.D-1)
        self.area_lu = (area_pu * (self.flow.units.characteristic_length_lu/
                                  self.flow.units.characteristic_length_pu)
                        ** (self.flow.stencil.d-1))
        self.nan_tensor = self.context.convert_to_tensor(torch.nan)
        self.solid_mask = self.context.convert_to_tensor(self.solid_mask, dtype=bool)

    def __call__(self, f: torch.Tensor = None):
        rho_tmp = torch.where(self.solid_mask, self.nan_tensor, self.flow.rho(f))
        rho_mean = torch.nanmean(rho_tmp)

        # get current force on obstacle in x direction
        force_x_lu = self.obstacle_boundary.force_sum[0]

        # calculate drag_coefficient in LU
        drag_coefficient = force_x_lu / (0.5 * rho_mean * self.flow.units.characteristic_velocity_lu ** 2 * self.area_lu)
        return drag_coefficient


class LiftCoefficient(Observable):
    """The coefficient of lift of an obstacle, calculated using momentum exchange method (MEM, MEA)
        - calculates the density,
        - gets the force in y direction on the obstacle boundary,
        - calculates the coefficient of lift
    """

    def __init__(self, flow, obstacle_boundary, solid_mask, area_pu):
        super().__init__(flow)
        self.obstacle_boundary = obstacle_boundary
        self.solid_mask = solid_mask

        # cross-sectional area of obstacle in LU
        # (! mind length-dimension in 2D -> area-dimension = self.lattice.D-1)
        self.area_lu = (area_pu * (self.flow.units.characteristic_length_lu/
                                  self.flow.units.characteristic_length_pu)
                        ** (self.flow.stencil.d-1))
        self.nan_tensor = self.context.convert_to_tensor(torch.nan)
        self.solid_mask = self.context.convert_to_tensor(self.solid_mask, dtype=bool)

    def __call__(self, f: torch.Tensor = None):
        rho_tmp = torch.where(self.solid_mask, self.nan_tensor, self.flow.rho(f))
        rho_mean = torch.nanmean(rho_tmp)

        # get current force on obstacle in y direction
        force_y_lu = self.obstacle_boundary.force_sum[1]

        # calculate lift_coefficient in LU
        lift_coefficient = force_y_lu / (0.5 * rho_mean * self.flow.units.characteristic_velocity_lu ** 2 * self.area_lu)
        return lift_coefficient