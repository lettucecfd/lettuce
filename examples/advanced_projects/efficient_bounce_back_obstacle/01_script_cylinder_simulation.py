
# this file should contain all the MP2 stuff for low RE (basically a reworked MP1)

####################
# IMPORT

import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)

import sys
import warnings
import os
import psutil
import shutil
import resource

import matplotlib.pyplot as plt

import time
import datetime

from pyevtk.hl import imageToVTK

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# LETTUCE RELATED
import lettuce as lt
from obstacle_cylinder import ObstacleCylinder
from ebb_simulation import EbbSimulation
from reporter_ProfileReporter import ProfileReporter
from reporter_advanced_vtk_reporter import VTKReporterAdvanced, VTKsliceReporter
from observables_force_coefficients import DragCoefficient, LiftCoefficient

# AUX. CODE
from helperCode import Logger
from data_processing_and_plotting import plot_force_coefficient, analyze_periodic_timeseries, draw_circular_mask, ProfilePlotter


####################
# ARGUMENT PARSING: this script is supposed to be called with arguments, detailing all simulation- and system-parameters

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

# context and I/O
parser.add_argument("--name", default="cylinder_lowRe", help="name of the simulation, appears in output directory name")
parser.add_argument("--default_device", default="cuda", type=str, help="run on cuda or cpu")
parser.add_argument("--float_dtype", default="float64", choices=["float32", "float64", "single", "double", "half"], help="data type for floating point calculations in torch")
parser.add_argument("--t_sim_max", default=(72*60*60), type=float, help="max. walltime [s] to run the simulationn(); default is 72 h. simulation will stops at 0.99*t_max_sim; IMPORTANT: this whole scipt may take longer to execute, depending on I/O etc.")

parser.add_argument("--text_output_only", action='store_true', help="if you don't want pngs etc. to open, please use this flag; data is still saved to files") ##FORMER: --cluster
parser.add_argument("--no_data", action='store_true', help="set, if you want no directories created and no date saved. Only direct output")
parser.add_argument("--outdir", default=os.getcwd(), type=str, help="directory to save output files to; default is CWD")
parser.add_argument("--outdir_data", default=None, type=str, help="directory to save large/many files to; if not set, everything os saved to outdir")

# flow physics and geometry
parser.add_argument("--reynolds_number", default=200, type=float, help="Reynolds number")
parser.add_argument("--mach_number", default=0.1, type=float, help="Mach number (should stay < 0.3, and < 0.1 for highest accuracy. low Ma can lead to instability because of round of errors ")
parser.add_argument("--char_velocity_pu", default=1, type=float, help="characteristic velocity of the flow in physical units (PU)")

parser.add_argument("--char_length_lu", default=1, type=int, help="characteristic length of the flow in lattice units. Number of gridpoints per diameter for a circular cylinder")
parser.add_argument("--char_length_pu", default=1, type=float, help="characteristic length of the flow in physical units. Diameter of the cylinder in PU")
parser.add_argument("--domain_length_x_in_d", default=None, type=float, help="domain length in x-direction (direction of flow) in number of cylinder-diameters")
parser.add_argument("--domain_height_y_in_d", default=None, type=float,help="domain height in y-direction (orthogonal to flow and cylinder axis) in number of cylinder-diameters")
parser.add_argument("--domain_width_z_in_d", default=None, type=float,help="domain width in z-direction (orthogonal to flow, parallel to cylinder axis) in number of cylinder-diameters; IMPORTANT: if not set, 2D-Simulation is performed")

parser.add_argument("--perturb_init", action='store_true', help="perturb initial velocity profile to trigger vortex shedding")
parser.add_argument("--u_init_condition", default=0, type=int, help="initial velocity field: # 0: uniform u=0, # 1: uniform u=1, # 2: parabolic, amplitude u_char_lu (similar to poiseuille-flow)")
parser.add_argument("--lateral_walls", default='periodic', help="OPTIONS: 'periodic' or 'bounceback'; add lateral walls, converting the flow to a cylinder in a channel. The velocity profile will be adjusted to be parabolic!")

# LBM solver settings
parser.add_argument("--n_steps", default=100000, type=int, help="number of steps to simulate, overwritten by t_target, if t_target is >0")
parser.add_argument("--t_target", default=0, type=float, help="time in PU to simulate, overwrites n_steps if t_target > 0")
parser.add_argument("--collision", default="bgk", type=str, choices=["kbc", "bgk", "reg", 'reg', "bgk_reg", 'kbc', 'bgk', 'bgk_reg'], help="collision operator (bgk, kbc, reg)")
parser.add_argument("--stencil", default="D3Q27", choices=['D2Q9', 'D3Q15', 'D3Q19', 'D3Q27'], help="stencil (D2Q9, D3Q27, D3Q19, D3Q15), IMPORTANT: should match number of dimensions inferred from domain_width! Otherwise default D2Q9 or D3Q27 will be chosen for 2D and 3D respectively")
parser.add_argument("--eqlm", action="store_true", help="use Equilibium LessMemory to save ~20% on GPU VRAM, sacrificing ~2% performance")
#TODO: how to use EQLM in lettuce 2025?
parser.add_argument("--bbbc_type", default='fwbb', help="bounce back algorithm (fwbb, hwbb, ibb1) for the solid obstacle")

# reporter and observable settings
parser.add_argument("--periodic_region_start_relative", default=None, type=float, help="RELATIVE (0.0-1.0) assumed start of the periodic region for measurement of temporal and spacial averaging of observables (drag, lift , velocity profiles...)")
parser.add_argument("--periodic_region_start_pu", default=None, type=float, help="ABSOLUTE PU-time; assumed start of the periodic region for measurement of temporal and spacial averaging of observables (drag, lift , velocity profiles...)")
parser.add_argument("--periodic_region_start_lu", default=None, type=int, help="ABSOLUTE LU-steps; assumed start of the periodic region for measurement of temporal and spacial averaging of observables (drag, lift , velocity profiles...)")

parser.add_argument("--calc_u_profiles", action='store_true', help="calculate average velocity profiles similar to [Di Ilio et al. 2018] and output plots and time-averages data for plots")
parser.add_argument("--output_u_profiles_timeseries", default=False, help="output average velocity profiles over time (full timeseries)")
parser.add_argument("--profile_reference_path", default="../profile_reference_data/", type=str, help="path to reference profiles from [Di Ilio et al. 2018]")

parser.add_argument("--vtk_full_basic", action='store_true', help="output vtk files of full domain each interval steps")
parser.add_argument("--vtk_full_basic_interval", type=int, help="step interval for output of basic full vtk files")

parser.add_argument("--vtk_3D", action='store_true', help="output vtk files of full domain each interval steps between start and end (if start and end are defined!)")
parser.add_argument("--vtk_3D_fps", type=float)
parser.add_argument("--vtk_3D_step_interval", type=float)
parser.add_argument("--vtk_3D_t_interval", type=float)
parser.add_argument("--vtk_3D_step_start", type=int)
parser.add_argument("--vtk_3D_step_end", type=int)
parser.add_argument("--vtk_3D_t_start", type=float)
parser.add_argument("--vtk_3D_t_end", type=float)

parser.add_argument("--vtk_slice2D", action='store_true', help="toggle vtk-output of 2D slice of WHOLE DOMAIN (!) to outdir_data, if set True (1)")
parser.add_argument("--vtk_slice2D_fps", type=float)
parser.add_argument("--vtk_slice2D_step_interval", type=float)
parser.add_argument("--vtk_slice2D_t_interval", type=float)
parser.add_argument("--vtk_slice2D_step_start", type=int)
parser.add_argument("--vtk_slice2D_step_end", type=int)
parser.add_argument("--vtk_slice2D_t_start", type=float)
parser.add_argument("--vtk_slice2D_t_end", type=float)

#TODO (optional): add NAN reporter (on/off, interval,...)
#TODO (optional): add watchdog reporter (on/off, interval,...)
#TODO (optional): add highMa reporter (on/off, interval,...)

#TODO (optional): add 2D-mp4-reporter... (fps_video, number of frames OR fps_pu)

# Checkpointing
# TODO (optional): add checkpointing-utilities (read, write):
''' Checkpointing is NOT implemented in lettuce 2025 at the moment.
It was present in lettuce 0.2.3
Relevant parameters and functionalities would/could include:
 - checkpoint IN path (where to READ a checkpoint from)
 - checkpoint OUT path (where to WRITE a checkpoint to)
 - checkpoint i_start, or t_start (which timestep (in LU or PU) 
 does the checkpoint correspond to.
 - should stuff continue on, or start from 0? (observables, time (step number),
 statistics (mean, max, min of observables etc.)  
'''

# for debugging purposes:
parser.add_argument("--count_tensors", action="store_true", help="(for debugging: count all tensors, their sizes and memory consumption on the GPU, to find memory leaks")


###########################################################
# put arguments in dictionary
print("SCRIPT: Writing arguments to dictionary...")
args = vars(parser.parse_args())

# print all arguments
print(f"SCRIPT: Input arguments are: \n{args}\n")

###########################################################

# CREATE timestamp, sim-ID, outdir and outdir_data
print("SCRIPT: Creating timestamp, simulation ID and creating output directory...")
name = args["name"]
outdir = args["outdir"]
outdir_data = args["outdir_data"]

default_device = args["default_device"]
float_dtype = args["float_dtype"]
t_sim_max = args["t_sim_max"]

text_output_only = args["text_output_only"]
no_data_flag = args["no_data"]

timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
sim_id = str(timestamp) + "-" + name

os.makedirs(outdir+"/"+sim_id) # create output dir
print(f"outdir/simID = {outdir}/{sim_id}")

# save data to regular outdir, if data-dir is not specified
if outdir_data is None:
    outdir_data = outdir

# adding individal sim-ID to outdir path to get individual DIR per simulation
outdir = outdir+"/"+sim_id
outdir_data = outdir_data+"/"+sim_id
if not os.path.exists(outdir_data):
    # create output dir for large/many files, if specified
    os.makedirs(outdir_data)
print(f"outdir_DATA/simID = {outdir}/{sim_id}")

# save input arguments/parameters to file in outdir:
print(f"SCRIPT: Writing input parameters to file: {outdir}/input_parameters.txt")
output_file = open(outdir+"/input_parameters.txt", "a")
for key in args:
    output_file.write('{:30s} {:30s}\n'.format(str(key), str(args[key])))
output_file.close()

### SAVE SCRIPT: save this script to outdir
print(f"SCRIPT: Saving simulation script to outdir...")
temp_script_name = sim_id + "_" + os.path.basename(__file__)
shutil.copy(__file__, outdir+"/"+temp_script_name)
print(f"-> Saved simulation script to '{str(outdir+'/'+temp_script_name)}'")

# START LOGGER -> get all terminal output into file
print(f"SCRIPT: Starting stdout-LOGGER (see outdir for log file)")
old_stdout = sys.stdout
sys.stdout = Logger(outdir)

###########################################################

# PROCESS AND SET PARAMETERS
print(f"SCRIPT: Processing parameters...")

# calculate domain and obstacle geometry and infer dimensions (2D, 3D)
if args["domain_height_y_in_d"] is None or args["domain_height_y_in_d"] <= 1:
    domain_height_y_in_d = 3
else:
    domain_height_y_in_d = args["domain_height_y_in_d"]

if args["domain_length_x_in_d"] is None or args["domain_length_x_in_d"] <=1 :
    # D/X = domain length in X- / flow-direction
    domain_length_x_in_d = 2 * domain_height_y_in_d
else:
    domain_length_x_in_d = args["domain_length_x_in_d"]

if args["domain_width_z_in_d"] is None:  # will be 2D
    dims = 2
    domain_width_z_in_d = None
else: # will be 3D
    dims = 3
    if args["domain_width_z_in_d"] <= 1/args["char_length_lu"] :
        # if less than 1 lattice node... ->set to 1 lattice node
        domain_width_z_in_d = 1/args["char_length_lu"]
        print("(!) domain_width_z_in_d is less than 1 lattice node: "
              "setting domain_width_z_in_d to 1 lattice node")
    else:
        domain_width_z_in_d = args["domain_width_z_in_d"]

# CORRECT GPT for symmetry:
'''
if D/Y (height of the domain in number of cylinder diameters) is even, 
the resulting GPD (gridpoints per diemeter) can't be odd for a symmetrical 
cylinder and symmetrical domain!
 - if D/Y is even, GPD will be corrected to be even for symmetrical cylinder
 => use odd D/Y value to use an odd GPD value!
'''
gpd_correction = False
if domain_height_y_in_d % 2 == 0 and args["char_length_lu"] % 2 != 0:
    gpd_correction = True  # gpd will be corrected
    gpd_setup = args["char_length_lu"]  # store old gpd for output
    char_length_lu = int(gpd_setup / 2) * 2  # make gpd even
    print("(!) domain_height_y_ind_d (DpY) is even, "
          "gridpoints per diameter (GPD, char_length_lu) will be set to"
          + str(char_length_lu) + ". Use odd domain_height_Y_in_D (DpY) "
                                  "to enable use of odd GPD (char_length_lu)!")
else:
    char_length_lu = args["char_length_lu"]

char_length_pu = args["char_length_pu"]
char_velocity_pu = args["char_velocity_pu"]
reynolds_number = args["reynolds_number"]
mach_number = args["mach_number"]

perturb_init = args["perturb_init"]
u_init_condition = args["u_init_condition"]

# calculate lu-domain-resolution,
# ...total number of gridpoints and check correct stencil
if dims == 2:
    resolution = [int(domain_length_x_in_d * char_length_lu),
                  int(domain_height_y_in_d * char_length_lu)]
    number_of_gridpoints = char_length_lu ** 2 * domain_length_x_in_d * domain_height_y_in_d
    if args["stencil"] == "D2Q9":
        stencil = lt.D2Q9()
    else:
        print("(!) WARNING: wrong stencil choice for 2D simulation, D2Q9 is used")
        stencil= lt.D2Q9()
else:
    resolution = [int(domain_length_x_in_d * char_length_lu),
                  int(domain_height_y_in_d * char_length_lu),
                  int(domain_width_z_in_d * char_length_lu)]
    number_of_gridpoints = (char_length_lu ** 3 * domain_length_x_in_d
                            * domain_height_y_in_d * domain_width_z_in_d)
    if args["stencil"] == "D3Q15":
        stencil = lt.D3Q15()
    elif args["stencil"] == "D3Q19":
        stencil = lt.D3Q19()
    elif args["stencil"] == "D3Q27":
        stencil = lt.D3Q27()
    else:
        print("(!) WARNING: wrong stencil choice for 3D simulation, D3Q27 is used")
        stencil = lt.D3Q27()

# read dtype
if float_dtype == "float32" or float_dtype == "single":
    float_dtype = torch.float32
elif float_dtype == "double" or float_dtype == "float64":
    float_dtype = torch.float64
elif float_dtype == "half" or float_dtype == "float16":
    float_dtype = torch.float16

# OVERWRITE n_steps, if t_target is given
n_steps = args["n_steps"]
t_target = args["t_target"]
if args["t_target"] > 0:
    n_steps = int(t_target * (char_length_lu / char_length_pu)
                  * (char_velocity_pu / (mach_number * 1 / np.sqrt(3))))
else:
    t_target = n_steps / (char_length_lu/char_length_pu
                          * char_velocity_pu/(mach_number*1/np.sqrt(3)))

# calculate relative starting point for observables that are calculated over a
# ...temporally converged or periodic region:
'''
- the temporal beginning of the periodic region depends on the time the flow 
needs to converge into - for example - a von Karman vortex street. 
 => This time is also dependent on the domain size!
EXAMPLE: For an Re=100 in a reasonable domain size, the flow needs 
         t_PU = 75-100 seconds to reach it's periodic state
'''
#TODO (optional): change periodic_start parameter to abs. step-number, instead of relative start:
# - ease of use for reporters and processing
# - no round-off error in preocessing LU-parameter, if set
if args["periodic_region_start_lu"] is not None and args["periodic_region_start_lu"] >= 0:
    periodic_start = args["periodic_region_start_lu"] / n_steps
elif args["periodic_region_start_pu"] is not None and args["periodic_region_start_pu"] >= 0:
    periodic_start = args["periodic_region_start_pu"] / t_target
elif (args["periodic_region_start_relative"] is not None and
      0 <= args["periodic_region_start_relative"] < 1.0):
    periodic_start = args["periodic_region_start_relative"]
else:
    if args["reynolds_number"] > 1000:
        periodic_start = 0.4
    else:
        periodic_start = 0.9

# check EQLM parameter
if args["eqlm"]:
    # TODO: use EQLM ( QuadraticEquilibriumLessMemory() ) how is this used in new lettuce?
    pass

# print temporal parameters: steps, T_PU
print(f"\n(INFO) parameters set for simulation of {n_steps} steps, "
      f"representing {t_target:.3f} seconds [PU]!\n")

###########################################
# INITIALIZE SOLVER COMPONENTS

print("SCRIPT: initializing solver components...")

# CONTEXT
print("-> initializing context...")
context = lt.Context(device=default_device, dtype=float_dtype,use_native=False)

# FLOW
print("-> initializing flow...")
flow = ObstacleCylinder(context=context, resolution=resolution, stencil=stencil,
                        reynolds_number=reynolds_number, mach_number=mach_number,
                        char_length_pu=char_length_pu, char_length_lu=char_length_lu,
                        char_velocity_pu=char_velocity_pu,
                        bc_type=str(args["bbbc_type"]), calc_force_coefficients=True,
                        lateral_walls=args["lateral_walls"])

# COLLISION OPERATOR
print("-> initializing collision operator...")
collision_operator = None
if args["collision"].casefold() == "reg" or args["collision"].casefold() == "bgk_reg":
    collision_operator = lt.RegularizedCollision(tau=flow.units.relaxation_parameter_lu)
elif args["collision"].casefold() == "kbc":
    # if dims == 2:
    #     collision_operator = lt.KBCCollision2D(tau=flow.units.relaxation_parameter_lu)
    # else:
    #     collision_operator = lt.KBCCollision3D(tau=flow.units.relaxation_parameter_lu)
    collision_operator = lt.KBCCollision(tau=flow.units.relaxation_parameter_lu)
else:  # default to bgk
    collision_operator = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu)

# SIMULATION
print("\nSCRIPT: initializing simulation object...")
simulation = EbbSimulation(flow, collision_operator,reporter=[])


# REPORTERS: initialize and append to simulation.reporter
print("-> initializing reporters...")

# DRAG and LIFT (Force Coefficients) and respective reporters:
cylinder_cross_sectional_area = flow.char_length_pu if dims==2 else (
        flow.char_length_pu*domain_width_z_in_d)

DragObservable = DragCoefficient(flow, simulation.post_streaming_boundaries[-1],
                                 solid_mask=simulation.post_streaming_boundaries[-1].mask,
                                 area_pu=cylinder_cross_sectional_area)
DragReporter = lt.ObservableReporter(DragObservable, interval=1, out=None)
simulation.reporter.append(DragReporter)

LiftObservable = LiftCoefficient(flow, simulation.post_streaming_boundaries[-1],
                                 solid_mask=simulation.post_streaming_boundaries[-1].mask,
                                 area_pu=cylinder_cross_sectional_area)
LiftReporter = lt.ObservableReporter(LiftObservable, interval=1, out=None)
simulation.reporter.append(LiftReporter)

# VELOCITY- and REYNOLDS-STRESS profiles over space and time:
if args["calc_u_profiles"]:
    # define positions
    position_1 = flow.x_pos_lu - 0.5 + 1.06 * flow.radius_lu * 2
    position_2 = flow.x_pos_lu - 0.5 + 1.54 * flow.radius_lu * 2
    position_3 = flow.x_pos_lu - 0.5 + 2.02 * flow.radius_lu * 2
    print("ProfileReporters at x-positions:" + " p1: " + str(position_1) + " p2:  " + str(
        position_2) + " p3:  " + str(position_3))

    # create and append profileReporters
    ProfileReporter1 = ProfileReporter(flow=flow, interval=1, position_lu=position_1,
                                       i_start= int(n_steps * periodic_start))
    simulation.reporter.append(ProfileReporter1)
    ProfileReporter2 = ProfileReporter(flow=flow, interval=1, position_lu=position_2,
                                       i_start= int(n_steps * periodic_start))
    simulation.reporter.append(ProfileReporter2)
    ProfileReporter3 = ProfileReporter(flow=flow, interval=1, position_lu=position_3,
                                       i_start= int(n_steps * periodic_start))
    simulation.reporter.append(ProfileReporter3)

# NAN REPORTER (if nan detected -> stop simulation)
# TODO (optional): add NaN reporter (see PR for that topic)
# - NEEDS breakable simulation (!) -> while loop. see ISSUE/Pull-request...

# HighMa Reporter (if Ma>0.3 detected -> report and/or stop simulation)
# TODO (optional): HighMa Reporter

# Watchdog/Progress-Reporter
# TODO (optional): Progress-Reporter

# VTK Reporter: for visualization etc.

# BASIC lettuce vtk output (see also: advanced vtk reporter below)
if args["vtk_full_basic"]:
    vtk_reporter = lt.VTKReporter(interval=args["vtk_full_basic_interval"],
                                  filename_base=outdir_data+"/vtk/out")

    # export obstacle mask for visualization
    mask_dict = dict()
    if dims ==2:
        mask_dict["mask"] = flow.obstacle_mask[...,None].astype(int)
    else:
        mask_dict["mask"] = flow.obstacle_mask.astype(int)
    imageToVTK(
        path=outdir_data+"/vtk/obstacle_point",
        pointData=mask_dict,
    )
    imageToVTK(
        path=outdir_data+"/vtk/obstacle_cell",
        cellData=mask_dict,
    )

    simulation.reporter.append(vtk_reporter)


# ADVANCED vtk output
# 3D
if args["vtk_3D"] and dims == 3:
    if args["vtk_3D_t_start"] is not None:
        #print("(vtk) overwriting vtk_step_start with {}, because vtk_t_start = {}")
        vtk_3d_i_start = int(round(flow.units.convert_time_to_lu(args["vtk_3D_t_start"])))
    elif args["vtk_3D_step_start"] is not None:
        vtk_3d_i_start = int(args["vtk_3D_step_start"])
    else:
        vtk_3d_i_start = 0

    if args["vtk_3D_t_end"] is not None:
        #print("(vtk) overwriting vtk_step_end with {}, because vtk_t_end = {}")
        vtk_3d_i_end = int(flow.units.convert_time_to_lu(args["vtk_3D_t_end"]))
    elif args["vtk_3D_step_end"] is not None:
        vtk_3d_i_end = args["vtk_3D_step_end"]
    else:
        vtk_3d_i_end = n_steps

    if args["vtk_3D_t_interval"] is not None and args["vtk_3D_t_interval"] > 0:
        vtk_3d_interval = int(flow.units.convert_time_to_lu(args["vtk_3D_t_interval"]))
    elif args["vtk_3D_step_interval"] is not None and args["vtk_3D_step_interval"] > 0:
        vtk_3d_interval = args["vtk_3D_step_interval"]
    elif args["vtk_3D_fps"] is not None and args["vtk_3D_fps"] > 0:
        vtk_3d_interval = int(flow.units.convert_time_to_lu(1 / args["vtk_3D_fps"]))
    else:
        vtk_3d_interval = 1

    if vtk_3d_interval < 1:
        vtk_3d_interval = 1

    vtk_3d_reporter = VTKReporterAdvanced(flow,
                                  interval=int(vtk_3d_interval),
                                  filename_base=outdir_data + "/vtk/out",
                                  imin=vtk_3d_i_start, imax=vtk_3d_i_end)
    simulation.reporter.append(vtk_3d_reporter)
    vtk_3d_reporter.output_mask(flow.solid_mask, outdir_data + "/vtk", "solid_mask",
                                point=True)


# slice2D (2D slice at Z/2)
if args["vtk_slice2D"]:
    if args["vtk_slice2D_t_start"] is not None and args["vtk_slice2D_t_start"] > 0:
        #print("(vtk) overwriting vtk_step_start with {}, because vtk_t_start = {}")
        vtk_slice2d_i_start = int(round(flow.units.convert_time_to_lu(args["vtk_slice2D_t_start"])))
    elif args["vtk_slice2D_step_start"] is not None and args["vtk_slice2D_step_start"] > 0:
        vtk_slice2d_i_start = int(args["vtk_slice2D_step_start"])
    else:
        vtk_slice2d_i_start = 0

    if args["vtk_slice2D_t_end"] is not None and args["vtk_slice2D_t_end"] > 0:
        #print("(vtk) overwriting vtk_step_end with {}, because vtk_t_end = {}")
        vtk_slice2d_i_end = int(flow.units.convert_time_to_lu(args["vtk_slice2D_t_end"]))
    elif args["vtk_slice2D_step_end"] is not None and args["vtk_slice2D_step_end"] > 0:
        vtk_slice2d_i_end = int(args["vtk_slice2D_step_end"])
    else:
        vtk_slice2d_i_end = n_steps

    if args["vtk_slice2D_t_interval"] is not None and args["vtk_slice2D_t_interval"] > 0:
        vtk_slice2d_interval = int(flow.units.convert_time_to_lu(args["vtk_slice2D_t_interval"]))
    elif args["vtk_slice2D_step_interval"] is not None and args["vtk_slice2D_step_interval"] > 0:
        vtk_slice2d_interval = int(args["vtk_slice2D_step_interval"])
    elif args["vtk_slice2D_fps"] is not None and args["vtk_slice2D_fps"] > 0:
        vtk_slice2d_interval = int(flow.units.convert_time_to_lu(1 / args["vtk_slice2D_fps"]))
    else:
        vtk_slice2d_interval = 1

    if vtk_slice2d_interval < 1:
        vtk_slice2d_interval = 1

    if args["vtk_slice2D"]:
        vtk_domainSlice_reporter = VTKsliceReporter(flow,
                               interval=int(vtk_slice2d_interval),
                               filename_base=outdir_data + "/vtk/slice_domain/slice_domain",
                               sliceXY=([0,resolution[0]-1],[0,resolution[1]-1]),
                               sliceZ=int(resolution[2]/2),
                               imin=vtk_slice2d_i_start, imax=vtk_slice2d_i_end)
        simulation.reporter.append(vtk_domainSlice_reporter)


# DRAW CYLINDER-MASK in 2D (xy-plane)
try:
    draw_circular_mask(flow, flow.char_length_lu, filebase=outdir_data, output_data=True)
except:
    print("(!) Drawing of circular cylinder mask did not work...")

##################################################
# PRINT relevant PARAMETERS prior to simulation:
print(f"\nSCRIPT: spacial and temporal dimensions:")
print("domain shape (LU):", flow.resolution)
print("t_target (PU) with", n_steps, "steps (LU):",
      round(n_steps * (flow.char_length_pu / flow.char_length_lu)
            * (mach_number * 1 / np.sqrt(3) / flow.char_velocity_pu), 3),
      "seconds")
print("steps to simulate 1 second PU:",
      round((flow.char_length_lu / flow.char_length_pu)
            * (flow.char_velocity_pu / (mach_number * 1 / np.sqrt(3))), 3), "steps")
print(f"steps to simulate {t_target:.3f} (t_target, PU) seconds: {t_target 
       * round((flow.char_length_lu / flow.char_length_pu) 
               * (flow.char_velocity_pu / (mach_number * 1 / np.sqrt(3))), 3):.3f} steps")

##################################################
# RUN SIMULATION:
print(f"\n#################################################")
print(f"\nSCRIPT ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}): "
      f"running simulation for {n_steps} steps...\n")
print(f"#################################################\n")

t_start = time.time()
mlups = simulation(num_steps=n_steps)
t_end = time.time()
runtime = t_end - t_start

print(f"***** SIMULATION FINISHED AT "
      f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} *****\n")

##################################################
# OUTPUT STATS:
print(f"### STATS ###")
print(f"MLUPS: {mlups:.3f}")
print(f"simulated PU-Time: {flow.units.convert_time_to_pu(n_steps)} seconds")
print("simulated LU-steps: ", n_steps)
print(f"runtime (WALLTIME) of simulation(num_steps): "
      f"{runtime:.3f} seconds (= ", round(runtime / 60, 2), "minutes )")
print("")
# GPU VRAM
print("### HARDWARE UTILIZATION ###")
print(f"current GPU VRAM (MB) usage: "
      f"{torch.cuda.memory_allocated(device=context.device)/1024/1024:.3f}")
print(f"max. GPU VRAM (MB) usage: "
      f"{torch.cuda.max_memory_allocated(device=context.device)/1024/1024:.3f}")
# CPU
[cpuLoad1, cpuLoad5, cpuLoad15] = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
print("CPU % avg. over last 1 min, 5 min, 15 min; ",
      round(cpuLoad1, 2), round(cpuLoad5, 2), round(cpuLoad15, 2))
# RAM
ram = psutil.virtual_memory()
print("current total RAM usage [MB]: " + str(round(ram.used / (1024 * 1024), 2)) + " of " + str(
    round(ram.total / (1024 * 1024), 2)) + " MB")

### export stats
if not no_data_flag:
    output_file = open(outdir + "/stats.txt", "a")
    output_file.write("DATA for " + timestamp)
    output_file.write("\n\n###   SIM-STATS  ###")
    output_file.write(
        f"\nruntime: {runtime:.3f} seconds (= {round(runtime / 60, 2)} "
        f"min = {round(runtime / 3600, 2)} h)")
    output_file.write("\nMLUPS = " + str(mlups))
    output_file.write("\n")
    output_file.write("\nVRAM_current [MB] = " + str(
        torch.cuda.memory_allocated(context.device) / 1024 / 1024))
    output_file.write("\nVRAM_peak [MB] = " + str(
        torch.cuda.max_memory_allocated(context.device) / 1024 / 1024))
    output_file.write("\n")
    output_file.write("\nCPU load % avg. over last 1, 5, 15 min: " + str(
        round(cpuLoad1, 2)) + " %, " + str(round(cpuLoad5, 2)) + " %, " + str(
        round(cpuLoad15, 2)) + " %")
    output_file.write("\nCurrent total RAM usage [MB]: " + str(
        round(ram.used / (1024 * 1024), 2)) + " of " + str(
        round(ram.total / (1024 * 1024), 2)) + " MB")
    output_file.write("\nmaximum total RAM usage ('MaxRSS') [MB]: " + str(
        round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              2)) + " MB")
    output_file.close()

##################################################
# PLOTTING 2D IMAGE of VELOCITY
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
fig.subplots_adjust(right=0.85)
u = flow.u_pu.cpu().numpy()
if dims == 2: # 2D
    im1 = axes[0].imshow(context.convert_to_ndarray(flow.solid_mask.T), origin="lower")
    im2 = axes[1].imshow(u[0, ...].T, origin="lower")
else: # 3D
    im1 = axes[0].imshow(context.convert_to_ndarray(flow.solid_mask[:, :,
            int(flow.solid_mask.shape[2] / 2)].T), origin="lower")
    im2 = axes[1].imshow(u[0, ...][:, :, int(flow.solid_mask.shape[2] / 2)].T, origin="lower")
cbar_ax = fig.add_axes((0.88, 0.15, 0.04, 0.7))
fig.colorbar(im2, cax=cbar_ax)
plt.show()
#TODO (optional): plot image of pressure

##################################################
# PROCESS DATA: calculate and SAVE OBSERVABLES AND PLOTS:
print("\nSCRIPT: processing, plotting and saving data...\n")

# DRAG
drag_timeseries = np.array(np.array(DragReporter.out))
plot_force_coefficient(drag_timeseries, ylabel="Coefficient of Drag $C_{D}$",
                       ylim=(0.5, 1.6),
                       secax_functions_tuple=(flow.units.convert_time_to_lu,
                                              flow.units.convert_time_to_pu),
                       filenamebase=outdir_data+"/drag", periodic_start=periodic_start,
                       adjust_ylim=True)
drag_stats = analyze_periodic_timeseries(drag_timeseries, periodic_start_rel=0.5,
                                         name="drag", verbose=True,
                                         pu_per_step=flow.units.convert_time_to_pu(1),
                                         outdir=outdir_data)

print(f"DRAG STATS:") #\n{drag_stats}")
for key, value in drag_stats.items():
    print(f"{key:<20} = {str(value)}")
print("")
'''
reminder:
 STATS ARE: {"mean_simple": mean_simple,
             "mean_periodcorrected": mean_periodcorrected,
             "min_simple": min_simple,
             "max_simple": max_simple,
             "max_mean": max_mean,
             "min_mean": min_mean,
             "frequency_fit": frequency_fit,
             "frequency_fft": freq_peak,
             "fft_resolution": freq_res}
'''

# LIFT
lift_timeseries = np.array(np.array(LiftReporter.out))
plot_force_coefficient(lift_timeseries, ylabel="Coefficient of Lift$C_{L}$", ylim=(-1.1, 1.1),
                       secax_functions_tuple=(flow.units.convert_time_to_lu,
                                              flow.units.convert_time_to_pu),
                       filenamebase=outdir_data+"/lift", periodic_start=periodic_start)
#OLD: lift_prominence = ((abs(lift_timeseries[2].max()) - abs(lift_timeseries[2].min())) * 0.5)
lift_stats = analyze_periodic_timeseries(lift_timeseries, periodic_start_rel=0.5,
                                         name="lift",
                                         verbose=True, pu_per_step=flow.units.convert_time_to_pu(1),
                                         outdir=outdir_data)
print(f"LIFT STATS:")
for key, value in lift_stats.items():
    print(f"{key:<20} = {str(value)}")

# plot DRAG and LIFT together:
try:
    fig, ax = plt.subplots(layout="constrained")
    drag_ax = ax.plot(drag_timeseries[:, 1], drag_timeseries[:, 2], color="tab:blue",
                      label="Drag")
    ax.set_xlabel("physical time / s")
    ax.set_ylabel("Coefficient of Drag $C_{D}$")
    ylim_adjusted = (drag_timeseries[int(drag_timeseries.shape[0] * periodic_start - 1):, 2].min()
                     * 0.5, drag_timeseries[int(drag_timeseries.shape[0] * periodic_start - 1):,
                     2].max() * 1.2)
    ax.set_ylim(ylim_adjusted)

    secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu,
                                                 flow.units.convert_time_to_pu))
    secax.set_xlabel("timesteps (simulation time / LU)")

    ax2 = ax.twinx()
    lift_ax = ax2.plot(lift_timeseries[:, 1], lift_timeseries[:, 2], color="tab:orange",
                       label="Lift")
    ax2.set_ylabel("Coefficient of Lift $C_{L}$")
    ax2.set_ylim((-1.1, 1.1))

    fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)

    if not no_data_flag:
        try:
            plt.savefig(outdir_data + "/dragAndLift_coefficient.png")
        except:
            print("(!) saving dragAndLift_coefficient.png didn't work!")
    plt.show()
except:
    print("(!) plotting drag and lift together didn't work!")

# STROUHAL number
# f = Strouhal for St=f*D/U and D=U=1 in PU
print("Strouhal number is: ", lift_stats["frequency_fit"]
      * flow.char_length_pu/flow.char_velocity_pu)

# AVERAGE VELOCITY and REYNOLDS STRESS PROFILES
if args["calc_u_profiles"] and args["profile_reference_path"] is not None:
    profile_plotter = ProfilePlotter(flow, output_path=outdir_data,
                                     reference_data_path=args["profile_reference_path"],
                                     i_timeseries=ProfileReporter1.i_out,
                                     u_timeseries1=ProfileReporter1.out,
                                     u_timeseries2=ProfileReporter2.out,
                                     u_timeseries3=ProfileReporter3.out)

    profile_plotter.process_data()
    profile_plotter.plot_velocity_profiles(show_reference=True, save=True)
    profile_plotter.plot_reynolds_stress_profiles(show_reference=True, save=True)


# EXPORT OBSERVABLES:
if not no_data_flag:
    ### CUDA-VRAM-summary:
    output_file = open(outdir_data +  "/" + timestamp + "_GPU_memory_summary.txt", "a")
    output_file.write("DATA for " + timestamp + "\n\n")
    output_file.write(torch.cuda.memory_summary(context.device))
    output_file.close()

    if args["count_tensors"]:
        try:
            ### list present torch tensors:
            output_file = open(outdir_data +  "/" + timestamp + "_GPU_list_of_tensors.txt", "a")
            total_bytes = 0
            import gc

            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        output_file.write("\n" + str(obj.size()) + ", "
                                          + str(obj.nelement() * obj.element_size()))
                        total_bytes = total_bytes + obj.nelement() * obj.element_size()
                except:
                    pass
            output_file.close()

            ### count occurence of tensors in list of tensors:
            from collections import Counter

            my_file = open(outdir_data +  "/" + timestamp + "_GPU_list_of_tensors.txt", "r")
            data = my_file.read()
            my_file.close()
            data_into_list = data.split("\n")
            c = Counter(data_into_list)
            output_file = open(outdir_data +  "/" + timestamp + "_GPU_counted_tensors.txt", "a")
            for k, v in c.items():
                output_file.write("type,size,bytes: {}, number: {}\n".format(k, v))
            output_file.write("\ntotal bytes for tensors:" + str(total_bytes))
            output_file.close()
        except:
            print("(!) counting tensors didn't work!")

############################################
# OUTPUT parameters, stats and observables
if not no_data_flag:
    output_file = open(outdir_data +  "/" + timestamp + "_parms_stats_obs.txt", "a")
    output_file.write("DATA for " + timestamp)
    output_file.write("\n\n###   SIM-Parameters   ###")
    output_file.write("\n{:30s} {:30s}".format("Re", str(reynolds_number)))
    output_file.write("\n{:30s} {:30s}".format("Ma", str(mach_number)))
    output_file.write("\n{:30s} {:30s}".format("n_steps", str(n_steps)))
    output_file.write("\n{:30s} {:30s}".format("t_target [s]", str(t_target)))
    output_file.write("\n{:30s} {:30s}".format("gridpoints_per_diameter (GPD)",
                                               str(flow.char_length_lu)))
    if gpd_correction:
        output_file.write("\n(!) gpd was corrected from: " + str(gpd_setup) + " to "
                          + str(flow.char_length_lu) + " because D/Y is even")
    output_file.write("\nDpX (D/X) = ".ljust(31) + str(domain_length_x_in_d))
    output_file.write("\nDpY (D/Y) = ".ljust(31) + str(domain_height_y_in_d))
    if flow.stencil.d == 3:
        output_file.write("\nDpZ (D/Z) = ".ljust(31) + str(domain_width_z_in_d))
    output_file.write("\nshape_LU: ".ljust(31) + str(flow.resolution))
    output_file.write(("\ntotal_number_of_gridpoints: ".ljust(31) + str(flow.rho(flow.f).numel())))
    output_file.write("\nbc_type = ".ljust(31) + str())
    output_file.write("\nlateral_walls = ".ljust(31) + str(flow.lateral_walls))
    output_file.write("\nstencil = ".ljust(31) + str(flow.stencil))
    output_file.write("\ncollision = ".ljust(31) + str(collision_operator))
    output_file.write("\n")
    # output_file.write("\nMa = " + str(mach_number))
    output_file.write("\nrelaxation parameter tau [LU]".ljust(31)
                      + str(flow.units.relaxation_parameter_lu))
    output_file.write("\ngrid_reynolds_number (Re_g) = ".ljust(31)
                      + str(flow.units.characteristic_velocity_lu/
                            ((1 / np.sqrt(3.0))**2 * (flow.units.relaxation_parameter_lu - 0.5))))
    output_file.write("\n")
    output_file.write("\ncylinder diameter PU = ".ljust(31)
                      + str(flow.units.characteristic_length_pu))
    output_file.write("\ncharacteristic velocity PU = ".ljust(31)
                      + str(flow.units.characteristic_velocity_pu))
    output_file.write("\nperturb_init = ".ljust(31) + str(perturb_init))
    output_file.write("\n")

    output_file.write("\n\n###   SIM-STATS  ###")
    output_file.write("\nruntime = ".ljust(31) + str(runtime)
                      + " seconds (=" + str(runtime / 60) + " minutes)")
    output_file.write("\nMLUPS = ".ljust(31) + str(mlups))
    output_file.write("\n")

    output_file.write("\nVRAM_current [MB] = ".ljust(31)
                      + str(torch.cuda.memory_allocated(context.device) / 1024 / 1024))
    output_file.write("\nVRAM_peak [MB] = ".ljust(31)
                      + str(torch.cuda.max_memory_allocated(context.device) / 1024 / 1024))
    output_file.write("\n")
    output_file.write("\nCPU load % avg. over last 1, 5, 15 min: ".ljust(31)
                      + str(round(cpuLoad1, 2)) + " %, " + str(round(cpuLoad5, 2))
                      + " %, " + str(round(cpuLoad15, 2)) + " %")
    output_file.write("\ntotal current RAM usage [MB]: ".ljust(31)
                      + str(round(ram.used / (1024 * 1024), 2)) + " of "
                      + str(round(ram.total / (1024 * 1024), 2)) + " MB")

    output_file.write("\n\n###   OBSERVABLES   ###")
    output_file.write("\nCoefficient of drag between "
                      + str(round(drag_timeseries[int(drag_timeseries.shape[0]
                                                      * periodic_start - 1), 1], 2)) + " s and "
                      + str(round(drag_timeseries[int(drag_timeseries.shape[0] - 1), 1], 2))
                      + " s:")
    output_file.write("\nCd_mean, simple      = ".ljust(31)
                      + str(drag_stats["mean_simple"]))
    output_file.write("\nCd_mean, peak_finder = ".ljust(31)
                      + str(drag_stats["mean_periodcorrected"]))
    output_file.write(
        "\nCd_min = ".ljust(31) + str(drag_stats["min_mean"] if drag_stats["min_mean"] is not None
                                      else drag_stats["min_simple"]))
    output_file.write(
        "\nCd_max = ".ljust(31) + str(drag_stats["max_mean"] if drag_stats["max_mean"] is not None
                                      else drag_stats["max_simple"]))
    output_file.write("\n")
    output_file.write("\nCoefficient of lift:")
    output_file.write("\nCl_min = ".ljust(31) + str(lift_stats["min_mean"]
                                                    if lift_stats["min_mean"] is not None
                                                    else lift_stats["min_simple"]))
    output_file.write("\nCl_max = ".ljust(31) + str(lift_stats["max_mean"]
                                                    if lift_stats["max_mean"] is not None
                                                    else lift_stats["max_simple"]))
    output_file.write("\n")
    output_file.write("\nStrouhal number:")
    output_file.write("\nFFT: St +- df = " + str(lift_stats["frequency_fft"]) + " +- "
                      + str(lift_stats["fft_resolution"]) + " Hz")
    output_file.write(
        "\nSINE-FIT: St = " + str(lift_stats["frequency_fit"]))
    output_file.write("\n")
    output_file.close()

# export flow physics to file:
output_file = open(outdir+"/flow_physics_parameters.txt", "a")
output_file.write('\n{:30s}'.format("FLOW PHYSICS and units:"))
output_file.write('\n')
output_file.write('\n{:30s} {:30s}'.format("Ma", str(reynolds_number)))
output_file.write('\n{:30s} {:30s}'.format("Re", str(mach_number)))
output_file.write('\n')
output_file.write('\n{:30s} {:30s}'.format("Relaxation Parameter LU",
                                           str(flow.units.relaxation_parameter_lu)))
output_file.write('\n{:30s} {:30s}'.format("l_char_LU",
                                           str(flow.units.characteristic_length_lu)))
output_file.write('\n{:30s} {:30s}'.format("u_char_LU",
                                           str(flow.units.characteristic_velocity_lu)))
output_file.write('\n{:30s} {:30s}'.format("viscosity_LU",
                                           str(flow.units.viscosity_lu)))
output_file.write('\n{:30s} {:30s}'.format("p_char_LU",
                                           str(flow.units.characteristic_pressure_lu)))
output_file.write('\n{:30s} {:30s}'.format("rho_char_LU",
                                           str(flow.units.characteristic_density_lu)))
output_file.write('\n')
output_file.write('\n{:30s} {:30s}'.format("l_char_PU",
                                           str(flow.units.characteristic_length_pu)))
output_file.write('\n{:30s} {:30s}'.format("u_char_PU",
                                           str(flow.units.characteristic_velocity_pu)))
output_file.write('\n{:30s} {:30s}'.format("viscosity_PU",
                                           str(flow.units.viscosity_pu)))
output_file.write('\n{:30s} {:30s}'.format("p_char_PU",
                                           str(flow.units.characteristic_pressure_pu)))
output_file.write('\n{:30s} {:30s}'.format("rho_char_PU",
                                           str(flow.units.characteristic_density_pu)))
output_file.write('\n')
output_file.write('\n{:30s} {:30s}'.format("grid reynolds number Re_g",
                                           str(flow.units.characteristic_velocity_lu/
                                               ((1 / np.sqrt(3.0))**2
                                                * (flow.units.relaxation_parameter_lu - 0.5)))))
output_file.write('\n{:30s} {:30s}'.format("flow through time PU [s]",
                                           str(domain_length_x_in_d
                                               *flow.units.characteristic_length_pu/
                                               flow.units.characteristic_velocity_pu)))
output_file.write('\n{:30s} {:30s}'.format("flow through time LU",
                                           str(domain_length_x_in_d
                                               *flow.units.characteristic_length_lu/
                                               flow.units.characteristic_velocity_lu)))
output_file.write('\n')
output_file.close()

## END OF SCRIPT
print(f"\n♬ THE END ♬")

# reset stdout (from LOGGER, see above (beginning))
sys.stdout = old_stdout

