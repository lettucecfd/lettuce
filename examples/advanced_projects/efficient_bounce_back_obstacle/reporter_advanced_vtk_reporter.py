import os
import numpy as np
import pyevtk.hl as vtk

from lettuce._simulation import Reporter, Simulation

__all__ = [
    "write_vtk", "VTKReporterAdvanced", "VTKsliceReporter"
]

def write_vtk(point_dict, id=0, filename_base="./data/output",
              origin: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    vtk.imageToVTK(
        path=f"{filename_base}_{id:08d}",
        origin=origin,
        spacing=(1.0, 1.0, 1.0),
        cellData=None,
        pointData=point_dict,
        fieldData=None,
    )

class VTKReporterAdvanced(Reporter):
    """
        Advanced VTK reporter. Adds functionality:
        - take solid_mask and pin observable values inside the solid to 0
            -> useful for suboptimal initialization or use of FWBB boundary
    """

    def __init__(self, flow, interval=50, filename_base="./data/output",
                 solid_mask=None, imin=0, imax=None):
        super().__init__(interval)
        self.flow = flow
        if interval < 0:
            self.interval = 1
        else:
            self.interval = interval
        self.filename_base = filename_base
        self.imin=imin
        if imax is None:
            self.imax = 1e15
        elif imax <= 0:
            self.imax = 1
        else:
            self.imax = imax
        if solid_mask is not None and flow.stencil.d == 2:
            self.solid_mask = solid_mask[..., None]
        else:
            self.solid_mask = solid_mask
        directory = os.path.dirname(filename_base)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        self.point_dict = dict()

    def __call__(self, simulation):
        if self.flow.i % self.interval == 0 and self.imin <= self.flow.i <= self.imax:
            u = self.flow.units.convert_velocity_to_pu(self.flow.u(self.flow.f))
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.flow.rho(self.flow.f))
#if you want density as well: rho = self.flow.units.convert_density_to_pu(self.flow.rho(f))
            if self.flow.stencil.d == 2:
                if self.solid_mask is None:
                    self.point_dict["p"] = self.flow.context.convert_to_ndarray(p[0, ..., None])
                else:
                    self.point_dict["p"] = np.where(self.solid_mask, 0,
                                            self.flow.context.convert_to_ndarray(p[0, ..., None]))
                for d in range(self.flow.stencil.d):
                    if self.solid_mask is None:
                        self.point_dict[f"u{'xyz'[d]}"] = (
                            self.flow.context.convert_to_ndarray(u[d, ..., None]))
                    else:
                        self.point_dict[f"u{'xyz'[d]}"] = (
                            np.where(self.solid_mask, 0,
                                     self.flow.context.convert_to_ndarray(u[d, ..., None])))
#if you want density as well: self.point_dict["rho"] = self.flow.context.convert_to_ndarray(rho[0, ..., None])
            else:
                if self.solid_mask is None:
                    self.point_dict["p"] = self.flow.context.convert_to_ndarray(p[0, ...])
                else:
                    self.point_dict["p"] = np.where(self.solid_mask, 0,
                                                self.flow.context.convert_to_ndarray(p[0, ...]))
                for d in range(self.flow.stencil.d):
                    if self.solid_mask is None:
                        self.point_dict[f"u{'xyz'[d]}"] = (
                            self.flow.context.convert_to_ndarray(u[d, ...]))
                    else:
                        self.point_dict[f"u{'xyz'[d]}"] = (
                            np.where(self.solid_mask, 0,
                                     self.flow.context.convert_to_ndarray(u[d, ...])))

#if you want density as well: self.point_dict["rho"] = self.flow.context.convert_to_ndarray(rho[0, ...])
            write_vtk(self.point_dict, self.flow.i, self.filename_base)

    def output_mask(self, mask, outdir=None, name="mask", point=False, no_offset=False):
        """
            output the mask as a vtk-file for visualizatione etc.

        UPDATE 28.08.2024 (MBille: outputs mask as cell data. cell data represents the approx.
            location of solid boundaries, assuming Fullway or Halfway Bounce Back implementation,
            if translated by (-0.5,-0.5,-0.5) LU.
            Attention: point data is misleading, looking at masks rendered as solid objects or
            point-clouds!
            USE: in Paraview use Filter:
                Threshold -> Above Upper Threshold (Upper Threshold 0.9)
                -> Solid Color -> Volume/Wireframe,...
        """

        if outdir is None:
            filename_base = self.filename_base
        else:
            filename_base = outdir+"/"+str(name)

        mask_dict = dict()

        mask_dict["mask"] = mask.astype(int) if len(mask.shape) == 3 \
            else mask[..., None].astype(int)
            # extension to pseudo-3D is needed for Paraview

        if point:
            vtk.imageToVTK(
                path=filename_base +"_point",
                pointData=mask_dict
            )
        if no_offset:
            vtk.imageToVTK(
                path=filename_base +"_cell_noOffset",
                cellData=mask_dict
            )
        vtk.imageToVTK(
            path=filename_base + "_cell",
            cellData=mask_dict,
            origin=(-0.5, -0.5, -0.5),
            spacing=(1.0, 1.0, 1.0)
        )
        
class VTKsliceReporter(Reporter):
    '''
        reports a certain specified area portion of the domain as vtk-file
    '''
    def __init__(self, flow, interval=50, filename_base="./data/output",
                 solid_mask=None, sliceXY=None, sliceZ=None, imin=0,
                 imax=None):
        super().__init__(interval)
        self.flow = flow
        self.interval = interval
        self.filename_base = filename_base
        self.imin = imin
        if imax is None:
            self.imax = 1e15
        elif imax <= 0:
            self.imax = 1
        else:
            self.imax = imax

        if solid_mask is not None and self.flow.stencil.d == 2:
            self.solid_mask = solid_mask[..., None]
        else:
            self.solid_mask = solid_mask

        directory = os.path.dirname(filename_base)
        if not os.path.isdir(directory):
            os.makedirs(directory)

        self.point_dict = dict()


        if sliceXY is not None:
            if sliceZ is None:
                sliceZ = 0
            self.xmin = sliceXY[0][0]
            self.ymin = sliceXY[1][0]
            self.xmax = sliceXY[0][1]
            self.ymax = sliceXY[1][1]
            self.z_index = sliceZ
        else:
            self.z_index = None
        # TODO (OPTIONAL):
        #  check if xmin == xmax (and y... and z...) and if so, add a "None"
        #  dimension and take only ONE value, so this becomes a slice.
        #  -> Probably best to do this in call() from a tuple:
        #       ([xmin, xmax],[ymin, ymax],[zmin, zmax])



    def __call__(self, simulation):
        if self.flow.i % self.interval == 0 and self.imin <= self.flow.i <= self.imax:
            if self.z_index is not None:
                f = self.flow.f[:,self.xmin:self.xmax+1, self.ymin:self.ymax+1, self.z_index, None]
                # (!) None is needed because single-slice omits the last dimension and will result
                # in bad dimension in conversion of u and rho/p below!
            else:
                f = self.flow.f
            u = self.flow.units.convert_velocity_to_pu(self.flow.u(f))
            p = self.flow.units.convert_density_lu_to_pressure_pu(self.flow.rho(f))
            # if you want the density: rho = self.flow.units.convert_density_to_pu(self.flow.rho(f))
            if self.flow.stencil.d == 2:
                if self.solid_mask is None:
                    self.point_dict["p"] = self.flow.context.convert_to_ndarray(p[0, ..., None])
                else:
                    self.point_dict["p"] = np.where(self.solid_mask, 0,
                                            self.flow.context.convert_to_ndarray(p[0, ..., None]))
                for d in range(self.flow.stencil.d):
                    if self.solid_mask is None:
                        self.point_dict[f"u{'xyz'[d]}"] = (
                            self.flow.context.convert_to_ndarray(u[d, ..., None]))
                    else:
                        self.point_dict[f"u{'xyz'[d]}"] = np.where(self.solid_mask, 0,
                                           self.flow.context.convert_to_ndarray(u[d, ..., None]))
            # if you want the density: self.point_dict["rho"] = self.flow.context.convert_to_ndarray(rho[0, ..., None])
            else:
                if self.solid_mask is None:
                    self.point_dict["p"] = self.flow.context.convert_to_ndarray(p[0, ...])
                else:
                    self.point_dict["p"] = np.where(self.solid_mask, 0,
                                                self.flow.context.convert_to_ndarray(p[0, ...]))
                for d in range(self.flow.stencil.d):
                    if self.solid_mask is None:
                        self.point_dict[f"u{'xyz'[d]}"] = (
                            self.flow.context.convert_to_ndarray(u[d, ...]))
                    else:
                        self.point_dict[f"u{'xyz'[d]}"] = np.where(self.solid_mask, 0,
                                               self.flow.context.convert_to_ndarray(u[d, ...]))

                # # if you want the density: self.point_dict["rho"] = self.flow.context.convert_to_ndarray(rho[0, ...])
            write_vtk(self.point_dict, self.flow.i, self.filename_base,
                      origin=(self.xmin, self.ymin, self.z_index))
                # origin added to show slice at correct position when superimposing on 3D vtk-data!

    def output_mask(self, mask, outdir=None, name="mask", point=False, no_offset=False):
        """
            output the mask as a vtk-file for visualizatione etc.

        UPDATE 28.08.2024 (MBille: outputs mask as cell data. cell data represents the approx.
            location of solid boundaries, assuming Fullway or Halfway Bounce Back implementation,
            if translated by (-0.5,-0.5,-0.5) LU.
            Attention: point data is misleading, looking at masks rendered as solid objects or
            point-clouds!
            USE: in Paraview use Filter:
                Threshold -> Above Upper Threshold (Upper Threshold 0.9)
                -> Solid Color -> Volume/Wireframe,...
        """

        if outdir is None:
            filename_base = self.filename_base
        else:
            filename_base = outdir + "/" + str(name)

        mask_dict = dict()

        mask_dict["mask"] = mask.astype(int) if len(mask.shape) == 3 else mask[..., None].astype(
            int)  # extension to pseudo-3D is needed for vtk-export to work

        if point:
            vtk.imageToVTK(
                path=filename_base + "_point",
                pointData=mask_dict
            )
        if no_offset:
            vtk.imageToVTK(
                path=filename_base + "_cell_noOffset",
                cellData=mask_dict
            )
        vtk.imageToVTK(
            path=filename_base + "_cell",
            cellData=mask_dict,
            origin=(-0.5, -0.5, -0.5),
            spacing=(1.0, 1.0, 1.0)
        )