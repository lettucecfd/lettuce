import numpy as np
import pyevtk.hl as vtk
import os

from ... import Reporter

__all__ = ['VTKReporter', 'write_vtk']


def write_vtk(point_dict, id=0, filename_base="./data/output"):
    vtk.gridToVTK(f"{filename_base}_{id:08d}",
                  np.arange(0, point_dict["p"].shape[0]),
                  np.arange(0, point_dict["p"].shape[1]),
                  np.arange(0, point_dict["p"].shape[2]),
                  pointData=point_dict)


class VTKReporter(Reporter):
    """General VTK Reporter for velocity and pressure"""

    def __init__(self, interval=50, filename_base="./data/output"):
        super().__init__(interval)
        self.filename_base = filename_base
        directory = os.path.dirname(filename_base)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        self.point_dict = dict()

    def __call__(self, simulation: 'Simulation'):
        if simulation.flow.i % self.interval == 0:
            u = simulation.flow.u_pu
            p = simulation.flow.p_pu
            if simulation.flow.stencil.d == 2:
                self.point_dict["p"] = (
                    simulation.flow.context.convert_to_ndarray(
                        p[0, ..., None]))
                for d in range(simulation.flow.stencil.d):
                    self.point_dict[f"u{'xyz'[d]}"] = (
                        simulation.flow.context.convert_to_ndarray(
                            u[d, ..., None]))
            else:
                self.point_dict["p"] = (
                    simulation.flow.context.convert_to_ndarray(p[0, ...]))
                for d in range(simulation.flow.stencil.d):
                    self.point_dict[f"u{'xyz'[d]}"] = (
                        simulation.flow.context.convert_to_ndarray(u[d, ...]))
            write_vtk(self.point_dict, simulation.flow.i, self.filename_base)

    def output_mask(self, simulation: 'Simulation'):
        """Outputs the no_collision_mask of the simulation object as VTK-file
        with range [0,1]
        Usage: vtk_reporter.output_mask(simulation.no_collision_mask)"""
        point_dict = dict()
        if simulation.flow.stencil.d == 2:
            point_dict["mask"] = simulation.flow.context.convert_to_ndarray(
                simulation.no_collision_mask)[..., None].astype(int)
        else:
            point_dict["mask"] = simulation.flow.context.convert_to_ndarray(
                simulation.no_collision_mask).astype(int)
        vtk.gridToVTK(self.filename_base + "_mask",
                      np.arange(0, point_dict["mask"].shape[0]),
                      np.arange(0, point_dict["mask"].shape[1]),
                      np.arange(0, point_dict["mask"].shape[2]),
                      pointData=point_dict)
