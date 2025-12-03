
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
import traceback

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import time
import datetime

from pyevtk.hl import imageToVTK

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# LETTUCE RELATED
import lettuce as lt
from obstacle_cylinder import ObstacleCylinder
from ebb_simulation import EbbSimulation
from observables_force_coefficients import DragCoefficient, LiftCoefficient

# AUX. CODE
from helperCode import Logger

# import pickle
# from copy import deepcopy
# from timeit import default_timer as timer
# from collections import Counter

#? warnings.simplefilter("ignore") # todo: is this needed?

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

#TODO additional physics and flow-parameters:
# - char_density (PU house)
# - char_pressure
# - vicsosity PU (house,

# LBM solver settings
parser.add_argument("--n_steps", default=100000, type=int, help="number of steps to simulate, overwritten by t_target, if t_target is >0")
parser.add_argument("--t_target", default=0, type=float, help="time in PU to simulate, overwrites n_steps if t_target > 0")
parser.add_argument("--collision", default="bgk", type=str, choices=["kbc", "bgk", "reg", 'reg', "bgk_reg", 'kbc', 'bgk', 'bgk_reg'], help="collision operator (bgk, kbc, reg)")
parser.add_argument("--stencil", default="D3Q27", choices=['D2Q9', 'D3Q15', 'D3Q19', 'D3Q27'], help="stencil (D2Q9, D3Q27, D3Q19, D3Q15), IMPORTANT: should match number of dimensions inferred from domain_width! Otherwise default D2Q9 or D3Q27 will be chosen for 2D and 3D respectively")
parser.add_argument("--eqlm", action="store_true", help="use Equilibium LessMemory to save ~20% on GPU VRAM, sacrificing ~2% performance")
#TODO: how to use EQLM in lettuce 2025?
parser.add_argument("--bbbc_type", default='fwbb', help="bounce back algorithm (fwbb, hwbb, ibb1, fwbbc, hwbbc2, ibb1c2) for the solid obstacle")

# reporter and observable settings
parser.add_argument("--vtk3D", action='store_true', help="output 3D vtk files")
#TODO: add vtk reporter (fps_pu, interval_lu,
# - 3D full
# - 2D slice normal (x,y,z)
# - obstacle point, obstacle cell speichern und abh. von 2D/3D korrekt formatieren
#TODO: add drag/lift reporter
# - periodic start relative (relative start of interval to do statistics over
# - plot lift, drag, strouhal number (try except...)
#TODO: add U-profile-reporter (output True, store/calculate true)
# -> condense in own functions
# -> make script to plot from data (see PLOTTING scripts in CYLINDER-paper for layout)
#TODO: add NAN reporter (on/off, interval)
#TODO: add watchdog reporter (on/off, interval)
#TODO: add highMa reporter (on/off, interval)
#TODO: add 2D-mp4-reporter... (fps_video, number of frames ODER fps_pu)

# Checkpointing
# TODO: add checkpointing-utilities (read, write)
# - checkpoint IN path
# - checkpoint OUT path (if only one is given, take this one for both)
# - checkpoint i_start, t_start?,


###########################################################
# OUTPUTS:
# - parameters INPUT
# - print 2D-slice with mask(s)
# - parameters SIMULATED (before sim())
# - 2D slice last frame
# - output condensed observables (drag, lift, strouhal) to file... @MP2
# - stats and performance (end)


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
if outdir_data is None: # save data to regular outdir, if data-dir is not specified
    outdir_data = outdir
outdir = outdir+"/"+sim_id  # adding individal sim-ID to outdir path to get individual DIR per simulation
outdir_data = outdir_data+"/"+sim_id
if not os.path.exists(outdir_data):
    os.makedirs(outdir_data) # create output dir for large/many files, if specified
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
# calc. relative starting point of peak_finding for Cd_mean Measurement to cut of any transients
#TODO: take absolute PU-time values here:
# - periodic_start_Re100_PU ~= 75-100 seconds
# - periodic_start hängt von Einschwingzeit ab! Diese hängt von der Domänengröße ab!
if args["reynolds_number"] > 1000:
    periodic_start = 0.4
else:
    periodic_start = 0.9

# calculate domain and obstacle geometry and infer dimensions (2D, 3D)
if args["domain_height_y_in_d"] is None or args["domain_height_y_in_d"] <= 1:
    domain_height_y_in_d = 3
else:
    domain_height_y_in_d = args["domain_height_y_in_d"]

if args["domain_length_x_in_d"] is None or args["domain_length_x_in_d"] <=1 :
    domain_length_x_in_d = 2 * domain_height_y_in_d # D/X = domain length in X- / flow-direction
else:
    domain_length_x_in_d = args["domain_length_x_in_d"]

if args["domain_width_z_in_d"] is None:  # will be 2D
    dims = 2
else: # will be 3D
    dims = 3
    if args["domain_width_z_in_d"] <= 1/args["char_length_lu"] : # if less than 1 lattice node
        domain_width_z_in_d = 1/args["char_length_lu"] # set to 1 lattice node
        print("(!) domain_width_z_in_d is less than 1 lattice node: setting domain_width_z_in_d to 1 lattice node")
    else:
        domain_width_z_in_d = args["domain_width_z_in_d"]

# CORRECT GPT for symmetry:
# if DpY is even, resulting GPD can't be odd for symmetrical cylinder and domain
# ...if DpY is even, GPD will be corrected to be even for symmetrical cylinder
# ...(!) use odd DpY to use odd GPD!
gpd_correction = False
if domain_height_y_in_d % 2 == 0 and args["char_length_lu"] % 2 != 0:
    gpd_correction = True  # gpd will be corrected
    gpd_setup = args["char_length_lu"]  # store old gpd for output
    char_length_lu = int(gpd_setup / 2) * 2  # make gpd even
    print("(!) domain_height_y_ind_d (DpY) is even, gridpoints per diameter (GPD, char_length_lu) will be set to" + str(
        char_length_lu) + ". Use odd domain_height_Y_in_D (DpY) to enable use of odd GPD (char_length_lu)!")
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
    resolution = [int(domain_length_x_in_d * char_length_lu), int(domain_height_y_in_d * char_length_lu)]
    number_of_gridpoints = char_length_lu ** 2 * domain_length_x_in_d * domain_height_y_in_d
    if args["stencil"] == "D2Q9":
        stencil = lt.D2Q9()
    else:
        print("(!) WARNING: wrong stencil choice for 2D simulation, D2Q9 is used")
        stencil= lt.D2Q9()
else:
    resolution = [int(domain_length_x_in_d * char_length_lu), int(domain_height_y_in_d * char_length_lu), int(domain_width_z_in_d * char_length_lu)]
    number_of_gridpoints = char_length_lu ** 3 * domain_length_x_in_d * domain_height_y_in_d * domain_width_z_in_d
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
    n_steps = int(t_target * (char_length_lu / char_length_pu) * (char_velocity_pu / (mach_number * 1 / np.sqrt(3))))
else:
    t_target = n_steps / (char_length_lu/char_length_pu * char_velocity_pu/(mach_number*1/np.sqrt(3)))

# check EQLM parameter
if args["eqlm"]:
    # TODO: use EQLM ( QuadraticEquilibriumLessMemory() ) how is this used in new lettuce?
    pass

print(f"\n(INFO) parameters set for simulation of {n_steps} steps, representing {t_target:.3f} seconds [PU]!\n")

###########################################

print("SCRIPT: initializing solver components...")

###
print("-> initializing context...")
context = lt.Context(device=default_device, dtype=float_dtype,use_native=False)

print("-> initializing flow...")
flow = ObstacleCylinder(context=context, resolution=resolution,
                        reynolds_number=reynolds_number, mach_number=mach_number,
                        char_length_pu=char_length_pu, char_length_lu=char_length_lu, char_velocity_pu=char_velocity_pu,
                        bc_type=str(args["bbbc_type"]), stencil=stencil, calc_force_coefficients=True, lateral_walls=args["lateral_walls"])

print("-> initializing collision operator...")
collision_operator = None
if args["collision"].casefold() == "reg" or args["collision"].casefold() == "bgk_reg":
    collision_operator = lt.RegularizedCollision(tau=flow.units.relaxation_parameter_lu)
elif args["collision"].casefold() == "kbc":
    if dims == 2:
        collision_operator = lt.KBCCollision2D(tau=flow.units.relaxation_parameter_lu)
    else:
        collision_operator = lt.KBCCollision3D(tau=flow.units.relaxation_parameter_lu)
else:  # default to bgk
    collision_operator = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu)

print("\nSCRIPT: initializing simulation object...")
simulation = EbbSimulation(flow, collision_operator,reporter=[])


# REPORTERS: initialize and append to simulation.reporter
print("-> initializing reporters...")

# DRAG and LIFT Force Coefficients and respective reporters:
cylinder_cross_sectional_area = flow.char_length_pu if dims==2 else flow.char_length_pu*domain_width_z_in_d
DragObservable = DragCoefficient(flow, simulation.post_streaming_boundaries[-1], solid_mask=simulation.post_streaming_boundaries[-1].mask, area_pu=cylinder_cross_sectional_area)
DragReporter = lt.ObservableReporter(DragObservable, interval=1, out=None)
simulation.reporter.append(DragReporter)
LiftObservable = LiftCoefficient(flow, simulation.post_streaming_boundaries[-1], solid_mask=simulation.post_streaming_boundaries[-1].mask, area_pu=cylinder_cross_sectional_area)
LiftReporter = lt.ObservableReporter(LiftObservable, interval=1, out=None)
simulation.reporter.append(LiftReporter)

# NAN REPORTER (if nan detected -> stop simulation)
# TODO: add NaN reporter
# - NEEDS breakable simulation (!) -> while loop. see ISSUE/Pull-request...

# HighMa Reporter (if Ma>0.3 retected -> report and/or stop simulation)
# TODO: HighMa Reporter

# Watchdog/Progress-Reporter
# TODO: Progress-Reporter

# VTK Reporter -> visualization
if args["vtk3D"]:
    #TODO: make stuff parameterized: interval, outdir, ...
    vtk_reporter = lt.VTKReporter(interval=1, filename_base=outdir_data+"/vtk/out")

    # export obstacle
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


# DRAW CYLINDER-MASK in 2D (xy-plane)
# TODO: port draw_circular_mask function from MP2/Paper to helperCode.py

##################################################
# PRINT PARAMETERS prior to simulation:
print(f"\nSCRIPT: spacial and temporal dimensions:")
print("domain shape (LU):", flow.resolution)
print("t_target (PU) with", n_steps, "steps (LU):",
      round(n_steps * (flow.char_length_pu / flow.char_length_lu) * (mach_number * 1 / np.sqrt(3) / flow.char_velocity_pu), 2),
      "seconds")
print("steps to simulate 1 second PU:",
      round((flow.char_length_lu / flow.char_length_pu) * (flow.char_velocity_pu / (mach_number * 1 / np.sqrt(3))), 2), "steps")
print("steps to simulate", t_target, " (t_target, PU) seconds:",
      t_target * round((flow.char_length_lu / flow.char_length_pu) * (flow.char_velocity_pu / (mach_number * 1 / np.sqrt(3))), 2),
      "steps")

##################################################
# RUN SIMULATION:
print(f"\n#################################################")
print(f"\nSCRIPT: running simulation for {n_steps} steps...")
print(f"#################################################\n")
t_start = time.time()
mlups = simulation(num_steps=n_steps)
t_end = time.time()
runtime = t_end - t_start

##################################################
# OUTPUT STATS:
print(f"### STATS ###")
print("MLUPS:", mlups)
print("simulated PU-Time:  ", flow.units.convert_time_to_pu(n_steps), " seconds")
print("simulated LU-steps: ", n_steps)
print("runtime (WALLTIME) of simulation(num_steps): ", runtime, "seconds (= ", round(runtime / 60, 2), "minutes )")
print("\n")
print("current VRAM (MB): ", torch.cuda.memory_allocated(context.device) / 1024 / 1024)
print("max. VRAM (MB): ", torch.cuda.max_memory_allocated(context.device) / 1024 / 1024)

[cpuLoad1, cpuLoad5, cpuLoad15] = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]
print("CPU % avg. over last 1 min, 5 min, 15 min; ", round(cpuLoad1, 2), round(cpuLoad5, 2), round(cpuLoad15, 2))

ram = psutil.virtual_memory()
print("current total RAM usage [MB]: " + str(round(ram.used / (1024 * 1024), 2)) + " of " + str(
    round(ram.total / (1024 * 1024), 2)) + " MB")

### export stats
if not no_data_flag:
    output_file = open(outdir_data + "/" + timestamp + "_stats.txt", "a")
    output_file.write("DATA for " + timestamp)
    output_file.write("\n\n###   SIM-STATS  ###")
    output_file.write("\nruntime = " + str(runtime) + " seconds (=" + str(runtime / 60) + " minutes)")
    output_file.write("\nMLUPS = " + str(mlups))
    output_file.write("\n")
    output_file.write("\nVRAM_current [MB] = " + str(torch.cuda.memory_allocated(context.device) / 1024 / 1024))
    output_file.write("\nVRAM_peak [MB] = " + str(torch.cuda.max_memory_allocated(context.device) / 1024 / 1024))
    output_file.write("\n")
    output_file.write("\nCPU load % avg. over last 1, 5, 15 min: " + str(round(cpuLoad1, 2)) + " %, " + str(round(cpuLoad5, 2)) + " %, " + str(round(cpuLoad15, 2)) + " %")
    output_file.write("\ntotal current RAM usage [MB]: " + str(round(ram.used / (1024 * 1024), 2)) + " of " + str(round(ram.total / (1024 * 1024), 2)) + " MB")
    output_file.close()

##################################################
# PLOTTING 2D IMAGE
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
fig.subplots_adjust(right=0.85)
u = flow.u_pu.cpu().numpy()
print("Max Velocity:", u.max())
if dims == 2:
    im1 = axes[0].imshow(context.convert_to_ndarray(flow.solid_mask.T), origin="lower")
    im2 = axes[1].imshow(u[0, ...].T, origin="lower")
elif dims == 3:
    im1 = axes[0].imshow(context.convert_to_ndarray(flow.solid_mask[:, :, int(flow.solid_mask.shape[2] / 2)].T), origin="lower")
    im2 = axes[1].imshow(u[0, ...][:, :, int(flow.solid_mask.shape[2] / 2)].T, origin="lower")
cbar_ax = fig.add_axes((0.88, 0.15, 0.04, 0.7))
fig.colorbar(im2, cax=cbar_ax)
fig.show()

##################################################
# PROCESS DATA: calculate and SAVE OBSERVABLES AND PLOTS:

# COEFF. OF DRAG
try:
    try:
        drag_coefficient = np.array(DragReporter.out)
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(drag_coefficient[:, 1], drag_coefficient[:, 2])
        ax.set_xlabel("physical time / s")
        ax.set_ylabel("Coefficient of Drag Cd")
        ax.set_ylim((0.5, 1.6))  # change y-limits
        secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
        secax.set_xlabel("timestep (simulation time / LU)")
    except:
        print("(!) drag_plotting didn't work...'")

    if not no_data_flag:
        try:
            plt.savefig(outdir_data + "/drag_coefficient.png")
        except:
            print("(!) saving 'drag_coefficient.png' failed!")
        try:
            np.savetxt(outdir_data + "/drag_coefficient.txt", drag_coefficient,
                       header="stepLU  |  timePU  |  Cd  FROM str(timestamp)")
        except:
            print("(!) saving drag_timeseries failed!")
    ax.set_ylim((drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2].min() * 0.5,
                 drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2].max() * 1.2))
    if not no_data_flag:
        try:
            plt.savefig(outdir_data + "/drag_coefficient_adjusted.png")
        except:
            print("(!) saving drag_coefficient_adjusted.png failed!")
    plt.close()
except:
    print("(!) analysing drag_coefficient didn't work'")

# peak finder: try calculating the mean drag coefficient from an integer number of periods, if a clear periodic signal is found
try:
    values = drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2]

    peaks_max = find_peaks(values, prominence=((values.max() - values.min()) / 2))
    peaks_min = find_peaks(-values, prominence=((values.max() - values.min()) / 2))
    if peaks_min[0].shape[0] - peaks_max[0].shape[0] > 0:
        peak_number = peaks_max[0].shape[0]
    else:
        peak_number = peaks_min[0].shape[0]

    if peaks_min[0][0] < peaks_max[0][0]:
        first_peak = peaks_min[0][0]
        last_peak = peaks_max[0][peak_number - 1]
    else:
        first_peak = peaks_max[0][0]
        last_peak = peaks_min[0][peak_number - 1]

    drag_mean = values[first_peak:last_peak].mean()
    drag_mean_simple = values.mean()

    print("Cd, simple mean:     ", drag_mean_simple)
    print("Cd, peak_finder mean:", drag_mean)

    # PLOT PEAK-FINDING:
    drag_stepsLU = drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 0]
    peak_max_y = values[peaks_max[0]]
    peak_max_x = drag_stepsLU[peaks_max[0]]
    peak_min_y = values[peaks_min[0]]
    peak_min_x = drag_stepsLU[peaks_min[0]]

    plt.plot(drag_stepsLU, values)
    plt.scatter(peak_max_x[:peak_number], peak_max_y[:peak_number])
    plt.scatter(peak_min_x[:peak_number], peak_min_y[:peak_number])
    plt.scatter(drag_stepsLU[first_peak], values[first_peak])
    plt.scatter(drag_stepsLU[last_peak], values[last_peak])
    plt.savefig(outdir_data +  "/drag_coefficient_peakfinder.png")
    peakfinder = True
except:  # if signal is not sinusoidal enough, calculate only simple mean value
    print(
        "peak-finding didn't work... probably no significant peaks visible (Re<46?), or periodic region not reached (T too small)")
    values = drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2]
    drag_mean_simple = values.mean()
    peakfinder = False
    print("Cd, simple mean:", drag_mean_simple)
try:
    plt.close()
except:
    pass

# LIFT COEFFICIENT
try:
    lift_coefficient = np.array(LiftReporter.out)
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(lift_coefficient[:, 1], lift_coefficient[:, 2])
    ax.set_xlabel("physical time / s")
    ax.set_ylabel("Coefficient of Lift Cl")
    ax.set_ylim((-1.1, 1.1))

    secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
    secax.set_xlabel("timesteps (simulation time / LU)")
    if not no_data_flag:
        try:
            plt.savefig(outdir_data + "/lift_coefficient.png")
        except:
            print("(!) saving lift_coefficient.png didn't work!")
        try:
            np.savetxt(outdir_data + "/lift_coefficient.txt", lift_coefficient,
                       header="stepLU  |  timePU  |  Cl  FROM str(timestamp)")
        except:
            print("(!) saving lift_timeline didn't work!")
    Cl_min = lift_coefficient[int(lift_coefficient[:, 2].shape[0] * periodic_start):, 2].min()
    Cl_max = lift_coefficient[int(lift_coefficient[:, 2].shape[0] * periodic_start):, 2].max()
    print("Cl_peaks: \nmin", Cl_min, "\nmax", Cl_max)
    plt.close()
except:
    print("(!) analysing lift_coefficient didn't work!")

#############################################
# plot DRAG and LIFT together:
try:
    fig, ax = plt.subplots(layout="constrained")
    drag_ax = ax.plot(drag_coefficient[:, 1], drag_coefficient[:, 2], color="tab:blue", label="Drag")
    ax.set_xlabel("physical time / s")
    ax.set_ylabel("Coefficient of Drag Cd")
    ax.set_ylim((0.5, 1.6))

    secax = ax.secondary_xaxis('top', functions=(flow.units.convert_time_to_lu, flow.units.convert_time_to_pu))
    secax.set_xlabel("timesteps (simulation time / LU)")

    ax2 = ax.twinx()
    lift_ax = ax2.plot(lift_coefficient[:, 1], lift_coefficient[:, 2], color="tab:orange", label="Lift")
    ax2.set_ylabel("Coefficient of Lift Cl")
    ax2.set_ylim((-1.1, 1.1))

    fig.legend(loc="upper left", bbox_to_anchor=(0, 1), bbox_transform=ax.transAxes)

    if not no_data_flag:
        try:
            plt.savefig(outdir_data + "/dragAndLift_coefficient.png")
        except:
            print("(!) saving dragAndLift_coefficient.png didn't work!")
            plt.close()
except:
    print("(!) plotting drag and lift together didn't work!")

#####################################################
# STROUHAL number: (only makes sense for Re>46 and if periodic state is reached)
try:
    ### prototyped fft for frequency detection and calculation of strouhal-number
    # ! Drag_frequency is 2* Strouhal-Freq. Lift-freq. is Strouhal-Freq.

    X = np.fft.fft(lift_coefficient[:, 2])  # fft result (amplitudes)
    N = len(X)  # number of freqs
    n = np.arange(N)  # freq index
    T = N * flow.units.convert_time_to_pu(1)  # total time measured (T_PU)
    freq = n / T  # frequencies (x-axis of spectrum)

    plt.figure()
    plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")  # plot spectrum |X|(f)
    plt.xlabel("Freq (Hz)")
    plt.ylabel("FFT Amplitude |X(freq)|")
    plt.xlim(0, 1)
    # print("max. Amplitude np.abx(X).max():", np.abs(X).max())   # for debugging
    plt.ylim(0, np.abs(X[:int(X.shape[0] * 0.5)]).max())  # ylim, where highes peak is on left half of full spectrum

    if not no_data_flag:
        plt.savefig(outdir_data + "/fft_Cl.png")

    freq_res = freq[1] - freq[0]  # frequency-resolution
    X_abs = np.abs(X[:int(X.shape[0] * 0.4)])  # get |X| Amplitude for left half of full spectrum
    freq_peak = freq[np.argmax(X_abs)]  # find frequency with the highest amplitude
    print("Frequency Peak:", freq_peak, "+-", freq_res, "Hz")
    # f = Strouhal for St=f*D/U and D=U=1 in PU
except:
    print("fft for Strouhal didn't work")
    freq_res = 0
    freq_peak = 0
plt.close()

# TODO: calc St from f when: f = Strouhal for St=f*D/U and D=U=1 in PU


# EXPORT OBSERVABLES:
if not no_data_flag:
    ### CUDA-VRAM-summary:
    output_file = open(outdir_data +  "/" + timestamp + "_GPU_memory_summary.txt", "a")
    output_file.write("DATA for " + timestamp + "\n\n")
    output_file.write(torch.cuda.memory_summary(context.device))
    output_file.close()

    # TODO: make this optional...
    try:
        ### list present torch tensors:
        output_file = open(outdir_data +  "/" + timestamp + "_GPU_list_of_tensors.txt", "a")
        total_bytes = 0
        import gc

        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    output_file.write("\n" + str(obj.size()) + ", " + str(obj.nelement() * obj.element_size()))
                    total_bytes = total_bytes + obj.nelement() * obj.element_size()
            except:
                pass
        # output_file.write("\n\ntotal bytes for tensors:"+str(total_bytes))
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

# TODO: cleanup of parms, stats, obs...
# output parameters, stats and observables
if not no_data_flag:
    output_file = open(outdir_data +  "/" + timestamp + "_parms_stats_obs.txt", "a")
    output_file.write("DATA for " + timestamp)
    output_file.write("\n\n###   SIM-Parameters   ###")
    output_file.write("\nRe = " + str(reynolds_number))
    output_file.write("\nn_steps = " + str(n_steps))
    output_file.write("\nT_target = " + str(flow.units.convert_time_to_pu(n_steps)) + " seconds")
    output_file.write("\ngridpoints_per_diameter (gpd) = " + str(flow.char_length_lu))
    if gpd_correction:
        output_file.write("\ngpd was corrected from: " + str(gpd_setup) + " to " + str(
            flow.char_length_lu) + " because D/Y is even")
    output_file.write("\nDpX (D/X) = " + str(domain_length_x_in_d))
    output_file.write("\nDpY (D/Y) = " + str(domain_height_y_in_d))
    if flow.stencil.d == 3:
        output_file.write("\nDpZ (D/Z) = " + str(domain_width_z_in_d))
    output_file.write("\nshape_LU: " + str(flow.resolution))
    output_file.write(("\ntotal_number_of_gridpoints: " + str(flow.rho(flow.f).numel())))
    output_file.write("\nbc_type = " + str())
#    output_file.write("\nlateral_walls = " + str(lateral_walls))
#    output_file.write("\nstencil = " + str(stencil_choice))
#    output_file.write("\ncollision = " + str(collision))
    output_file.write("\n")
    output_file.write("\nMa = " + str(mach_number))
    output_file.write("\ntau = " + str(flow.units.relaxation_parameter_lu))
#    output_file.write("\ngrid_reynolds_number (Re_g) = " + str(re_g))
    output_file.write("\n")
    output_file.write("\nsetup_diameter_PU = " + str(flow.char_length_lu))
    output_file.write("\nflow_velocity_PU = " + str(flow.char_length_lu))
#    output_file.write("\nu_init = " + str(u_init))
    output_file.write("\nperturb_init = " + str(perturb_init))
    output_file.write("\n")
#    output_file.write("\noutput_vtk = " + str(output_vtk))
#    output_file.write("\nvtk_fps = " + str(vtk_fps))

    output_file.write("\n\n###   SIM-STATS  ###")
    output_file.write("\nruntime = " + str(runtime) + " seconds (=" + str(runtime / 60) + " minutes)")
    output_file.write("\nMLUPS = " + str(mlups))
    output_file.write("\n")

    output_file.write("\nVRAM_current [MB] = " + str(torch.cuda.memory_allocated(context.device) / 1024 / 1024))
    output_file.write("\nVRAM_peak [MB] = " + str(torch.cuda.max_memory_allocated(context.device) / 1024 / 1024))
    output_file.write("\n")
    output_file.write("\nCPU load % avg. over last 1, 5, 15 min: " + str(round(cpuLoad1, 2)) + " %, " + str(round(cpuLoad5, 2)) + " %, " + str(round(cpuLoad15, 2)) + " %")
    output_file.write("\ntotal current RAM usage [MB]: " + str(round(ram.used / (1024 * 1024), 2)) + " of " + str(round(ram.total / (1024 * 1024), 2)) + " MB")

    output_file.write("\n\n###   OBSERVABLES   ###")
    output_file.write("\nCoefficient of drag between " + str(round(drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1), 1], 2)) + " s and " + str(round(drag_coefficient[int(drag_coefficient.shape[0] - 1), 1], 2)) + " s:")
    output_file.write("\nCd_mean, simple      = " + str(drag_mean_simple))
    if peakfinder:
        output_file.write("\nCd_mean, peak_finder = " + str(drag_mean))
    else:
        output_file.write("\nnoPeaksFound")
    output_file.write(
        "\nCd_min = " + str(drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2].min()))
    output_file.write(
        "\nCd_max = " + str(drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2].max()))
    output_file.write("\n")
    output_file.write("\nCoefficient of lift:")
    output_file.write("\nCl_min = " + str(Cl_min))
    output_file.write("\nCl_max = " + str(Cl_max))
    output_file.write("\n")
    output_file.write("\nStrouhal number:")
    output_file.write("\nSt +- df = " + str(freq_peak) + " +- " + str(freq_res) + " Hz")
    output_file.write("\n")
    output_file.close()


## END OF SCRIPT
print(f"\n♬ THE END ♬")

# reset stdout (from LOGGER, see above)
sys.stdout = old_stdout


################################################################
# script components below...

# Process arguments and set parameters
# - I/O: create timestamp, sim-ID, outdir (path) and outdir_data (path)
# - flow physics: char_velocity, re, ma, density, pressure, length, domain (resolution)
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

# plotting and post-processing
# - save data

# reset LOGGER (!)