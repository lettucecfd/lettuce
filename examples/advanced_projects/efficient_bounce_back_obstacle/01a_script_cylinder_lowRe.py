
# this file should contain all the MP2 stuff for low RE (basically a reworked MP1)

# IMPORT

import lettuce as lt
from .obstacle_cylinder import ObstacleCylinder
from .ebb_simulation import EbbSimulation

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import sys
import warnings

import numpy as np

## OLD from lettuce import (LettuceException)
## OLD from lettuce.unit import UnitConversion
## OLD from lettuce.util import append_axes
## OLD from lettuce.boundary import EquilibriumBoundaryPU, EquilibriumOutletP, AntiBounceBackOutlet
## OLD from lettuce.flows.obstacleCylinder import ObstacleCylinder

## OLD from lettuce.equilibrium import QuadraticEquilibrium_LessMemory

## OLD from lettuce.max import draw_circular_mask

import torch
import time
import datetime
import os
import psutil
import shutil
from pyevtk.hl import imageToVTK
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pickle
from copy import deepcopy
from timeit import default_timer as timer
from collections import Counter

warnings.simplefilter("ignore") # todo: is this needed?

# ARGUMENT PARSING: this scipt is supposed to be called with arguments, detailling all simulation- and system-parameters

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
parser.add_argument("--re", default=200, type=float, help="Reynolds number")
parser.add_argument("--ma", default=0.1, type=float, help="Mach number (should stay < 0.3, and < 0.1 for highest accuracy. low Ma can lead to instability because of round of errors ")
parser.add_argument("--char_velocity_pu", default=1, type=float, help="characteristic velocity of the flow in physical units (PU)")

parser.add_argument("--char_length_lu", default=1, type=int, help="characteristic length of the flow in lattice units. Number of gridpoints per diameter for a circular cylinder")
parser.add_argument("--char_length_pu", default=1, type=float, help="characteristic length of the flow in physical units. Diameter of the cylinder in PU")
parser.add_argument("--domain_length_x_in_d", default=None, help="domain length in x-direction (direction of flow) in number of cylinder-diameters")
parser.add_argument("--domain_height_y_in_d", default=None, help="domain height in y-direction (orthogonal to flow and cylinder axis) in number of cylinder-diameters")
parser.add_argument("--domain_width_z_in_d", default=None, help="domain width in z-direction (orthogonal to flow, parallel to cylinder axis) in number of cylinder-diameters; IMPORTANT: if not set, 2D-Simulation is performed")

parser.add_argument("--perturb_init", action='store_true', help="perturb initial velocity profile to trigger vortex shedding")
parser.add_argument("--u_init_condition", default=0, type=int, help="initial velocity field: # 0: uniform u=0, # 1: uniform u=1, # 2: parabolic, amplitude u_char_lu (similar to poiseuille-flow)")

# solver settings
parser.add_argument("--n_steps", default=100000, type=int, help="number of steps to simulate, overwritten by t_target, if t_target is >0, end of sim will be step_start+n_steps")
parser.add_argument("--t_target", default=0, type=float, help="time in PU to simulate, t_start will be calculated by PU/LU-conversion of step_start")
parser.add_argument("--collision", default="bgk", type=str, choices=["kbc", "bgk", "reg", 'reg', "bgk_reg", 'kbc', 'bgk', 'bgk_reg'], help="collision operator (bgk, kbc, reg)")
parser.add_argument("--stencil", default="D3Q27", choices=['D2Q9', 'D3Q15', 'D3Q19', 'D3Q27'], help="stencil (D2Q9, D3Q27, D3Q19, D3Q15), IMPORTANT: should match number of dimensions infered from domain_width! Otherwise default D2Q9 or D3Q27 will be chosen for 2D and 3D respectively")
#TODO: check dimension and stencil match, OR: issue warning, choose stencil matching domain width
parser.add_argument("--eqlm", action="store_true", help="use Equilibium LessMemory to save ~20% on GPU VRAM, sacrificing ~2% performance")
parser.add_argument("--bbbc_type", default='fwbb', help="bounce back algorithm (fwbb, hwbb, ibb1, fwbbc, hwbbc2, ibb1c2) for the solid obstacle")

# reporter settings
#TODO: add vtk reporter
#TODO: add drag reporter
#TODO: add lift reporter
#TODO: add NAN reporter
#TODO: add watchdog reporter
#TODO: add highMa reporter

# put arguments in dictionary
args = vars(parser.parse_args())

###########################################################

# CREATE timestamp, sim-ID, outdir and outdir_data
name = args["name"]
outdir = args["outdir"]
outdir_data = args["outdir_data"]

default_device = args["default_device"]
float_dtype = args["float_dtype"]
t_sim_max = args["t_sim_max"]

text_output_only = args["text_output_only"]
no_data_flag = args["no_data_flag"]

timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
sim_id = str(timestamp) + "-" + name

os.makedirs(outdir+"/"+sim_id) # create output dir
print(f"outdir/simID = {outdir}/{sim_id}")
if outdir_data is None: # save data to regular outdir, if data-dir is not specified
    outdir_data = outdir
outdir = outdir+"/"+sim_id  # adding individal sim-ID to outdir path to get individual DIR per simulation
outdir_data = outdir_data+"/"+sim_id
if not os.path.exists(outdir_data):
    os.makedirs(outdir_data) # create output dir for large/many files, if specified
    print(f"Outdir_DATA/simID = {outdir}/{sim_id}")

# print all arguments
print(f"Input arguments: {args}")

# save input arguments/parameters to file in outdir:
output_file = open(outdir+"/input_parameters.txt", "a")
for key in args:
    output_file.write('{:30s} {:30s}\n'.format(str(key), str(args[key])))
output_file.close()

### SAVE SCRIPT: save this script to outdir
print(f"\nSaving simulation script to outdir...")
temp_script_name = sim_id + "_" + os.path.basename(__file__)
shutil.copy(__file__, outdir+"/"+temp_script_name)
print(f"Saved simulation script to '{str(outdir+'/'+temp_script_name)}'")

# START LOGGER -> get all terminal output into file
old_stdout = sys.stdout
sys.stdout = Logger(outdir)

#####################################

# PROCESS AND SET PARAMETERS

# calc. relative starting point of peak_finding for Cd_mean Measurement to cut of any transients
if args["re"] > 1000:
    periodic_start = 0.4
else:
    periodic_start = 0.9

# calculate domain and obstacle geometry
if args["domain_height_y_in_d"] is None or args["domain_height_y_in_d"] <= 1:
    domain_height_y_in_d = 3
else:
    domain_height_y_in_d = args["domain_height_y_in_d"]

if args["domain_length_x_in_d"] is None or args["domain_length_x_in_d"] <=1 :
    domain_length_x_in_d = 2 * domain_height_y_in_d # D/X = domain length in X- / flow-direction
else:
    domain_length_x_in_d = args["domain_length_x_in_d"]

if args["domain_width_in_d"] is None:  # will be 2D
    dims = 2
else: # will be 3D
    dims = 3

    if args["domain_width_in_d"] <= 1/args["char_length_lu"] : # if less than 1 lattice node
        domain_width_z_in_d = 1/args["char_length_lu"] # set to 1 lattice node
        print("(!) setting domain_width_in_d to 1 lattice node")

# if DpY is even, resulting GPD can't be odd for symmetrical cylinder and domain
# ...if DpY is even, GPD will be corrected to be even for symmetrical cylinder
# ...use odd DpY to use odd GPD
gpd_correction = False
if domain_height_y_in_d % 2 == 0 and args["char_length_lu"] % 2 != 0:
    gpd_correction = True  # gpd will be corrected
    gpd_setup = args["char_length_lu"]  # store old gpd for output
    char_length_lu = int(gpd_setup / 2) * 2  # make gpd even
    print("(!) domain_height_y_ind_d (DpY) is even, gridpoints per diameter (GPD, char_length_lu) will be set to" + str(
        char_length_lu) + ". Use odd domain_height_in_D (DpY) to enable use of odd GPD (char_length_lu)!")
else:
    char_length_lu = args["char_length_lu"]

char_length_pu = args["char_length_pu"]
char_velocity_pu = args["char_velocity_pu"]
reynolds_number = args["reynolds_number"]
mach_number = args["mach_number"]

perturb_init = args["perturb_init"]
u_init_condition = args["u_init_condition"]

# calculate lu-domain-resolution, total number of gridpoints and check correct stencil
if dims == 2:
    resolution = [domain_length_x_in_d * char_length_lu, domain_height_y_in_d * char_length_lu]
    number_of_gridpoints = char_length_lu ** 2 * domain_length_x_in_d * domain_height_y_in_d
    if args["stencil"] == "D2Q9":
        stencil = lt.D2Q9()
    else:
        print("WARNING: wrong stencil choice for 2D simulation, D2Q9 is used")
        stencil= lt.D2Q9()
else:
    resolution = [domain_length_x_in_d * char_length_lu, domain_height_y_in_d * char_length_lu, domain_width_z_in_d * char_length_lu]
    number_of_gridpoints = char_length_lu ** 3 * domain_length_x_in_d * domain_height_y_in_d * domain_width_z_in_d
    if args["stencil"] == "D3Q15":
        stencil = lt.D3Q15()
    elif args["stencil"] == "D3Q19":
        stencil = lt.D3Q19()
    elif args["stencil"] == "D3Q27":
        stencil = lt.D3Q27()
    else:
        print("WARNING: wrong stencil choice for 3D simulation, D3Q27 is used")
        stencil = lt.D3Q27()

# read dtype
if float_dtype == "float32" or float_dtype == "single":
    float_dtype = torch.float32
elif float_dtype == "double" or float_dtype == "float64":
    float_dtype = torch.float64
elif float_dtype == "half" or float_dtype == "float16":
    float_dtype = torch.float16

# OVERWRITE n_steps, if t_target is given
T_target = 0
if args["t_target"] > 0:
    T_target = args["t_target"]
    n_steps = int(T_target * ((char_length_lu) / char_length_pu) * (char_velocity_pu / (mach_number * 1 / np.sqrt(3))))

# check EQLM parameter
if args["eqlm"]:
    # TODO: use EQLM ( QuadraticEquilibriumLessMemory() ) how is this used in new lettuce?
    pass

###########################################
#todo parameter (args) vtk_fps,
# lateral_walls = args["lateral_walls"]
# #vtk_fps = 10  # FramesPerSecond (PU) for vtk-output
# cuda_device = args["default_device"]
# #nan_reporter = args["nan_reporter"]

###
context = lt.Context(default_device)

flow = ObstacleCylinder(context=context, resolution=resolution,
                        reynolds_number=re, mach_number=ma,
                        char_length_pu=, char_length_lu=, char_velocity_pu=,
                        bc_type=, stencil=stencil, equilibrium=)

relaxation_parameter_tau = flow.units.relaxation_parameter_lu
collision_operator = lt.BGKCollision(relaxation_parameter_tau)

#TODO reporter

simulation = EbbSimulation(flow, collision_operator,reporter=[])


simulation(num_steps=n_steps)

# Process arguments and set parameters
# - I/O: create timestamp, sim-ID, outdir (path) and outdir_data (path)
# - flow physics: char_velocity, re, ma, density, presure, length, domain (resolution)
# - solver settings

# save input parameters to file

# save this script to outdir for later reference

# start logger

# DEFINE AUXILIARY methods (or load from other helper-file)


# SETUP SIMULATOR
# - stencil (infer dim from stencil), dtype, equilibrium,
# - calculate obstacle geometry and domain constraints

# save geometry input to file

# flow, collision,...classes

# (opt.) plot stuff (velocity, geometry etc.)

# initialize REPORTERS

# RUN SIMULATION

# print stats

# export stats

# plotting and post processing
# - save data

# reset lOGGER (!)