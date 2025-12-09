# THIS FILE CONTAINS THE UTILITIES TO PROCESS AND PLOT THE REPORTED DATA FROM THE CYLINDER-SIMULATION

import numpy as np
import matplotlib.pyplot as plt
from typing import Any
import traceback
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from collections import Counter


# DRAG COEFFICIENT
# LIFT COEFFICIENT
# DRAG+LIFT PNG
# STROUHAL
# U-profile

#######################################################################

# PLOT FORCE COEFFICIENTs (data-array [i, t, value], ylabel-string, ylim tupel[float,float], outdir (for png and .txt), save_timeseries
# - plot data with ylim, and ylabel
# - save png to outdir
# - save timeseries txt to outdir
# - plot data with adjusted ylim
# - save png to outdir

def plot_force_coefficient(data_array: np.ndarray, ylabel: str, ylim: tuple[float, float], secax_functions_tuple: tuple[Any,Any], filenamebase, save_timeseries = False, periodic_start=0, adjust_ylim=False):
    # PLOT
    try:
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(data_array[:, 1], data_array[:, 2])
        ax.set_xlabel("physical time / s")
        ax.set_ylabel(str(ylabel))
        ax.set_ylim(ylim)  # change y-limits
        secax = ax.secondary_xaxis('top', functions=secax_functions_tuple)
        secax.set_xlabel("timestep (simulation time / LU)")
    except Exception as e:
        print(f"(WARNING!) plotting of {ylabel} didn't work...")
        print("\n--- Python Stack Trace ---")
        full_trace = traceback.format_exc()
        print(full_trace)
        print("--------------------------\n")

    # SAVE PNG
    try:
        plt.savefig(filenamebase + ".png")
    except Exception as e:
        print(f"(WARNING!) saving of {ylabel} PLOT to {filenamebase}.png didn't work...")
        print("\n--- Python Stack Trace ---")
        full_trace = traceback.format_exc()
        print(full_trace)
        print("--------------------------\n")

    # PLOT with ylim ADJUSTED
    if adjust_ylim:
        try:
            fig, ax = plt.subplots(constrained_layout=True)
            ax.plot(data_array[:, 1], data_array[:, 2])
            ax.set_xlabel("physical time / s")
            ax.set_ylabel(str(ylabel))
            ylim_2 = ( data_array[int(data_array.shape[0] * periodic_start - 1):,2].min() * 0.5,
                       data_array[int(data_array.shape[0] * periodic_start - 1):,2].max() * 1.2)
            ax.set_ylim(ylim_2)  # change y-limits
            secax = ax.secondary_xaxis('top', functions=secax_functions_tuple)
            secax.set_xlabel("timestep (simulation time / LU)")
        except Exception as e:
            print(f"(WARNING!) plotting of {ylabel} didn't work...")
            print("\n--- Python Stack Trace ---")
            full_trace = traceback.format_exc()
            print(full_trace)
            print("--------------------------\n")

        # SAVE PNG
        try:
            plt.savefig(filenamebase + "_ylimadjusted.png")
        except Exception as e:
            print(f"(WARNING!) saving of {ylabel} PLOT to {filenamebase}_ylimadjusted.png didn't work...")
            print("\n--- Python Stack Trace ---")
            full_trace = traceback.format_exc()
            print(full_trace)
            print("--------------------------\n")

    # SAVE .txt timeseries
    #TODO: make this a seperate function...
    if save_timeseries:
        try:
            np.savetxt(filenamebase + ".txt", data_array,
                       header=f"stepLU  |  timePU  |  {ylabel}")
        except Exception as e:
            print(f"(WARNING!) saving of {ylabel} TIMESERIES to {filenamebase}.txt didn't work...")
            print("\n--- Python Stack Trace ---")
            full_trace = traceback.format_exc()
            print(full_trace)
            print("--------------------------\n")

# ANALYZE PERIODIC SIGNAL (verbose (plot peak-finding),
# - min, max, mean (simple + window-corrected) (über ganzzahlige Periodenzahl gemittelt)
# - RETURN: dictionary mit: Name, min, max, mean_simple, mean_n_periodic, mean-min, mean-max (v.a. für Cl, nach neuen Skripten)
# - print error if peak-finding didn't work
# - (opt. FFT, periodicity, frequency)
#   - FFT
#   - print FFT spectrum
#   - (!) NEW FREQUENCY detection from MP2/Paper

def analyze_periodic_timeseries(data_array: np.ndarray, periodic_start_rel: float,prominence: float = None, name: str = "periodic timeseries", outdir=None, pu_per_step = None, verbose=False):
    values_periodic = data_array[int(data_array.shape[0] * periodic_start_rel - 1):, 2]
    steps_LU_periodic = data_array[int(data_array.shape[0] * periodic_start_rel - 1):, 0]
    mean_periodcorrected = None
    max_mean = None
    min_mean = None
    if prominence is None:
        prominence = ((values_periodic.max() - values_periodic.min()) * 0.5)
        #The prominence value is up for debate;
        # The value given here only reliably catches all peaks,
        # if the signal is simple and periodically converged


    try:
        peaks_max = find_peaks(values_periodic, prominence=prominence) # drag-prominence: ((values.max() - values.min()) / 2); lift-prominence: (lift1100_1500[:,2].max()+lift1100_1500[:,2].min())/2); oder lift: lift_converged[:,2].max()*0.5
        peaks_min = find_peaks(-values_periodic, prominence=prominence)

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

        if verbose:
            peak_max_y = values_periodic[peaks_max[0]]
            peak_max_x = steps_LU_periodic[peaks_max[0]]
            peak_min_y = values_periodic[peaks_min[0]]
            peak_min_x = steps_LU_periodic[peaks_min[0]]

            plt.subplots(constrained_layout=True)
            plt.plot(steps_LU_periodic, values_periodic)
            plt.scatter(peak_max_x[:peak_number], peak_max_y[:peak_number])
            plt.scatter(peak_min_x[:peak_number], peak_min_y[:peak_number])
            plt.scatter(steps_LU_periodic[first_peak], values_periodic[first_peak])
            plt.scatter(steps_LU_periodic[last_peak], values_periodic[last_peak])

            max_mean = values_periodic[peaks_max[0]].mean()
            plt.axhline(y=max_mean, color="r", ls="--", lw=0.5)
            min_mean = values_periodic[peaks_min[0]].mean()
            plt.axhline(y=min_mean, color="r", ls="--", lw=0.5)
            if outdir is not None:
                plt.savefig(outdir + f"/{name}_peakfinder.png")

        mean_periodcorrected = values_periodic[first_peak:last_peak].mean()
    except Exception as e:
        print(f"(WARNING!) peak finding for {name} didn't work... See Python Stack Trace below:")
        print("\n--- Python Stack Trace ---")
        full_trace = traceback.format_exc()
        print(full_trace)
        print("--------------------------\n")

    mean_simple = values_periodic.mean()
    min_simple = values_periodic.min()
    max_simple = values_periodic.max()

    # sine-fit
    def sine_func(xx, a, b, c, d):
        return a * np.sin(2 * np.pi * b * xx + c) + d

    frequency_fit = None
    try:
        coefficients, values = curve_fit(sine_func, steps_LU_periodic, values_periodic, p0=(0.7, 0.2, 0.5, 0))
        fig, ax = plt.subplots(constrained_layout=True)
        if verbose:
            plt.plot(steps_LU_periodic, values_periodic, steps_LU_periodic,
                     sine_func(steps_LU_periodic, *coefficients))
            plt.legend(["timeseries", "sine-fit"])
            ax.set_xlabel("physical time / s")
            ax.set_ylabel("timeseries")
            # ax.set_ylim([-1, 1])
            if outdir is not None:
                plt.savefig(outdir + f"/{name}_sine-fit.png")
        frequency_fit = coefficients[1]
    except Exception as e:
        print(f"(WARNING!) sine-fitting for {name} didn't work...")
        print("\n--- Python Stack Trace ---")
        full_trace = traceback.format_exc()
        print(full_trace)
        print("--------------------------\n")

    # FFT
    freq_peak = None
    freq_res = None

    if pu_per_step is not None:
        try:
            X = np.fft.fft(values_periodic)  # fft result (amplitudes)
            N = len(X)  # number of freqs
            n = np.arange(N)  # freq index
            T = N * pu_per_step  # total time measured (T_PU)
            freq = n / T  # frequencies (x-axis of spectrum)

            if verbose:
                plt.figure()
                plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")  # plot spectrum |X|(f)
                plt.xlabel("Freq (Hz)")
                plt.ylabel("FFT Amplitude |X(freq)|")
                plt.xlim(0, 1)
                # print("max. Amplitude np.abx(X).max():", np.abs(X).max())   # for debugging
                plt.ylim(0, np.abs(X[:int(X.shape[0] * 0.5)]).max())  # ylim, where highes peak is on left half of full spectrum

            if outdir is not None:
                plt.savefig(outdir + f"/{name}_fft.png")

            freq_res = freq[1] - freq[0]  # frequency-resolution
            X_abs = np.abs(X[:int(X.shape[0] * 0.4)])  # get |X| Amplitude for left half of full spectrum
            freq_peak = freq[np.argmax(X_abs)]  # find frequency with the highest amplitude
            # print("Frequency Peak:", freq_peak, "+-", freq_res, "Hz")
            # f = Strouhal for St=f*D/U and D=U=1 in PU
        except Exception as e:
            print(f"(WARNING!) fft for {name} didn't work...")
            print("\n--- Python Stack Trace ---")
            full_trace = traceback.format_exc()
            print(full_trace)
            print("--------------------------\n")


    return {"mean_simple": mean_simple, "mean_periodcorrected": mean_periodcorrected, "min_simple": min_simple,
            "max_simple": max_simple, "max_mean": max_mean, "min_mean": min_mean,
            "frequency_fit": frequency_fit, "frequency_fft": freq_peak, "fft_resolution": freq_res}


def draw_circular_mask(flow, gridpoints_per_diameter, output_data=False,
                       filebase=".", print_data=False):
    ### calculate and export 2D obstacle_mask as .png
    grid_x = gridpoints_per_diameter + 2
    if output_data:
        output_file = open(filebase + "/obstacle_mask_info.txt", "a")
        output_file.write("GPD = " + str(gridpoints_per_diameter) + "\n")
    if print_data:
        print("GPD = " + str(gridpoints_per_diameter))
    # define radius and position for a symmetrical circular Cylinder-Obstacle
    radius_LU = 0.5 * gridpoints_per_diameter
    y_pos_LU = 0.5 * grid_x + 0.5
    x_pos_LU = y_pos_LU

    # get x,y,z meshgrid of the domain (LU)
    xyz = tuple(np.linspace(1, n, n) for n in (grid_x,
                                               grid_x))  # tupel of list indizes (1-n (non zero-based!))
    xLU, yLU = np.meshgrid(*xyz,
                           indexing='ij')  # meshgrid of x- and y- indizes -> * unpacks the tuple to be two values and now a tuple

    # define cylinder (LU) (circle)
    obstacle_mask_for_visualization = np.sqrt(
        (xLU - x_pos_LU) ** 2 + (yLU - y_pos_LU) ** 2) < radius_LU

    nx, ny = obstacle_mask_for_visualization.shape  # number of x- and y-nodes (Skalar)

    rand_mask = np.zeros((nx, ny),
                         dtype=bool)  # for all the solid nodes, neighboring fluid nodes
    rand_mask_f = np.zeros((flow.stencil.q, nx, ny),
                           dtype=bool)  # same, but including q-dimension
    rand_xq = []  # list of all x-values (incl. q-multiplicity)
    rand_yq = []  # list of all y-values (incl. q-multiplicity)

    a, b = np.where(
        obstacle_mask_for_visualization)  # np.array: list of (a) x-coordinates und (b) y-coordinates of the obstacle_mask_for_visualization
    # ...to iterate over all boudnary/object/wall nodes
    for p in range(0,
                   len(a)):  # for all True-ndoes in obstacle_mask_for_visualization
        for i in range(0,
                       flow.stencil.q):  # for all stencil directions c_i (lattice.stencil.e)
            try:  # try in case the neighboring cell does not exist (an f pointing out of the simulation domain)
                if not obstacle_mask_for_visualization[
                    a[p] + flow.stencil.e[i][0], b[p] + flow.stencil.e[
                        i][1]]:
                    # if neighbor in +(e_x, e_y; e is c_i) is False, we are on the object-surface (self True with neighbor False)
                    rand_mask[a[p], b[p]] = 1
                    rand_mask_f[flow.stencil.opposite[i], a[p], b[p]] = 1
                    rand_xq.append(a[p])
                    rand_yq.append(b[p])
            except IndexError:
                pass  # just ignore this iteration since there is no neighbor there
    rand_x, rand_y = np.where(rand_mask)  # list of all surface coordinates
    x_pos = sum(rand_x) / len(rand_x)  # x-coordinate of circle center
    y_pos = sum(rand_y) / len(rand_y)  # y-coordinate of circle center

    # calculate all radii and r_max and r_min
    r_max = 0
    r_min = gridpoints_per_diameter
    radii = np.zeros_like(rand_x,
                          dtype=float)  # list of all redii (without q-dimension) in LU
    for p in range(0, len(rand_x)):  # for all nodes
        radii[p] = np.sqrt(
            (rand_x[p] - x_pos) ** 2 + (rand_y[
                                            p] - y_pos) ** 2)  # calculate distance to circle center
        if radii[p] > r_max:
            r_max = radii[p]
        if radii[p] < r_min:
            r_min = radii[p]

    # calculate all radii (with q-multiplicity)
    radii_q = np.zeros_like(rand_xq, dtype=float)
    for p in range(0, len(rand_xq)):
        radii_q[p] = np.sqrt(
            (rand_xq[p] - x_pos) ** 2 + (rand_yq[p] - y_pos) ** 2)

    ### all relative radii in relation to gpd/2
    radii_relative = radii / (
            radius_LU - 0.5)  # (substract 0.5 because "true" boundary location is 0.5LU further out than node-coordinates)
    radii_q_relative = radii_q / (radius_LU - 0.5)

    # calc. mean rel_radius
    r_rel_mean = sum(radii_relative) / len(radii_relative)
    rq_rel_mean = sum(radii_q_relative) / len(radii_q_relative)

    ## AREA calculation
    area_theory = np.pi * (
                gridpoints_per_diameter / 2) ** 2  # area = pi*r² in LU²
    area = len(
        a)  # area in LU = number of nodes, because every node has a cell of 1LU x 1LU around it

    if output_data:
        output_file.write(
            "\nr_rel_mean: " + str(sum(radii_relative) / len(radii_relative)))
        output_file.write("\nrq_rel_mean: " + str(
            sum(radii_q_relative) / len(radii_q_relative)))
        output_file.write("\nr_rel_min: " + str(r_max / (radius_LU - 0.5)))
        output_file.write("\nr_rel_max: " + str(r_min / (radius_LU - 0.5)))
        output_file.write("\n\narea_rel: " + str(area / area_theory))

        output_file.write("\n\nradii: " + str(Counter(radii)))
        output_file.write("\nradii_q: " + str(Counter(radii_q)) + "\n\n")
        output_file.close()
    if print_data:
        print("area_rel: " + str(area / area_theory))

    ### PLOT Mask
    plt.figure()
    plt.imshow(obstacle_mask_for_visualization)
    # plt.xticks(np.arange(gridpoints_per_diameter + 2), minor=True)
    # plt.yticks(np.arange(gridpoints_per_diameter + 2), minor=True)
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymax, ymin = ax.get_ylim()
    if gridpoints_per_diameter >= 10:
        plt.xticks(np.arange(0, xmax, int(xmax / 10)))
        plt.yticks(np.arange(0, ymax, int(ymax / 10)))
    else:
        plt.xticks(np.arange(0, xmax, 1))
        plt.yticks(np.arange(0, ymax, 1))
    plt.title("GPD = " + str(gridpoints_per_diameter))
    ax.set_xticks(np.arange(-.5, xmax, 1), minor=True)
    ax.set_yticks(np.arange(-.5, ymax, 1), minor=True)

    # grid thickness, cicrle, node marker
    x, y = np.meshgrid(np.linspace(0, int(xmax), int(xmax + 1)),
                       np.linspace(0, int(ymax), int(ymax + 1)))
    if gridpoints_per_diameter < 30:
        ax.grid(which="minor", color="k", axis='both', linestyle='-',
                linewidth=2)
        circle = plt.Circle((xmax / 2 - 0.25, ymax / 2 - 0.25),
                            gridpoints_per_diameter / 2, color='r', fill=False,
                            linewidth=1)
        ax.add_patch(circle)
        plt.plot(x, y, marker='.', linestyle='', color="b", markersize=1)
    elif gridpoints_per_diameter < 70:
        ax.grid(which="minor", color="k", axis='both', linestyle='-',
                linewidth=1)
        circle = plt.Circle((xmax / 2 - 0.25, ymax / 2 - 0.25),
                            gridpoints_per_diameter / 2, color='r', fill=False,
                            linewidth=0.5)
        ax.add_patch(circle)
    elif gridpoints_per_diameter < 100:
        ax.grid(which="minor", color="k", axis='both', linestyle='-',
                linewidth=0.5)
    elif gridpoints_per_diameter < 150:
        ax.grid(which="minor", color="k", axis='both', linestyle='-',
                linewidth=0.25)

    if output_data:
        plt.savefig(filebase + "/obstacle_mask_GPD" + str(
            gridpoints_per_diameter) + ".png")
    if print_data:
        plt.show()
    else:
        plt.close()

# SNIP: MP2 fit sinewave to Cl for better frequency-measurement)
### FIT SINEWAVE to converged Cl to get better St-measurement
# 1. get converged lift-curve
# 2. fit sinewave with starting-freq 0.2
# 3. store freq

# PLOT FORCE COEFFICIENTs together! -> im Skript

# u-PROFILES

#######################################

### load reference data from diIlio_path:
# avg_u_start = 0.5
# diIlio_path = "../literature/DiIlio_2018/"
#
# # import reference data: (data is: first collumn Y/D, second column u_d/u_char)
# # ux
# p1_LS1993_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos1_LS1993.csv', delimiter=';')
# p2_LS1993_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos2_LS1993.csv', delimiter=';')
# p3_LS1993_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos3_LS1993.csv', delimiter=';')
#
# p1_KM2000_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos1_KM2000.csv', delimiter=';')
# p2_KM2000_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos2_KM2000.csv', delimiter=';')
# p3_KM2000_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos3_KM2000.csv', delimiter=';')
#
# p1_WR2008_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos1_WR2008.csv', delimiter=';')
# p2_WR2008_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos2_WR2008.csv', delimiter=';')
# p3_WR2008_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos3_WR2008.csv', delimiter=';')
#
# p1_DI2018_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos1_DI2018.csv', delimiter=';')
# p2_DI2018_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos2_DI2018.csv', delimiter=';')
# p3_DI2018_ux = np.genfromtxt(diIlio_path + 'Fig09_ux_profile_pos3_DI2018.csv', delimiter=';')
#
# # uy
# p1_LS1993_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos1_LS1993.csv', delimiter=';')
# p2_LS1993_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos2_LS1993.csv', delimiter=';')
# p3_LS1993_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos3_LS1993.csv', delimiter=';')
#
# p1_KM2000_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos1_KM2000.csv', delimiter=';')
# p2_KM2000_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos2_KM2000.csv', delimiter=';')
# p3_KM2000_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos3_KM2000.csv', delimiter=';')
#
# p1_WR2008_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos1_WR2008.csv', delimiter=';')
# p2_WR2008_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos2_WR2008.csv', delimiter=';')
# p3_WR2008_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos3_WR2008.csv', delimiter=';')
#
# p1_DI2018_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos1_DI2018.csv', delimiter=';')
# p2_DI2018_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos2_DI2018.csv', delimiter=';')
# p3_DI2018_uy = np.genfromtxt(diIlio_path + 'Fig10_uy_profile_pos3_DI2018.csv', delimiter=';')
#
# # uxux
# p1_DI2018_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos1_DI2018.csv', delimiter=';')
# p1_KM2000_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos1_KM2000.csv', delimiter=';')
# p1_R2016_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos1_R2016.csv', delimiter=';')
# p2_BM1994_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos2_BM1994.csv', delimiter=';')
# p2_DI2018_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos2_DI2018.csv', delimiter=';')
# p2_KM2000_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos2_KM2000.csv', delimiter=';')
# p2_LS1993_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos2_LS1993.csv', delimiter=';')
# p2_R2016_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos2_R2016.csv', delimiter=';')
# p3_DI2018_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos3_DI2018.csv', delimiter=';')
# p3_KM2000_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos3_KM2000.csv', delimiter=';')
# p3_R2016_uxux = np.genfromtxt(diIlio_path + 'Fig11_uxux_profile_pos3_R2016.csv', delimiter=';')
#
# # uyuy
# p1_DI2018_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos1_DI2018.csv', delimiter=';')
# p1_R2016_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos1_R2016.csv', delimiter=';')
# p2_BM1994_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos2_BM1994.csv', delimiter=';')
# p2_DI2018_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos2_DI2018.csv', delimiter=';')
# p2_LS1993_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos2_LS1993.csv', delimiter=';')
# p2_R2016_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos2_R2016.csv', delimiter=';')
# p3_DI2018_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos3_DI2018.csv', delimiter=';')
# p3_R2016_uyuy = np.genfromtxt(diIlio_path + 'Fig12_uyuy_profile_pos3_R2016.csv', delimiter=';')
#
# # uxuy
# p1_BM1994_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos1_BM1994.csv', delimiter=';')
# p1_DI2018_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos1_DI2018.csv', delimiter=';')
# p1_R2016_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos1_R2016.csv', delimiter=';')
# p2_BM1994_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos2_BM1994.csv', delimiter=';')
# p2_DI2018_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos2_DI2018.csv', delimiter=';')
# p2_LS1993_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos2_LS1993.csv', delimiter=';')
# p2_R2016_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos2_R2016.csv', delimiter=';')
# p3_BM1994_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos3_BM1994.csv', delimiter=';')
# p3_DI2018_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos3_DI2018.csv', delimiter=';')
# p3_R2016_uxuy = np.genfromtxt(diIlio_path + 'Fig13_uxuy_profile_pos3_R2016.csv', delimiter=';')
#
# # plot 2x (CO) FIGURES: 5 AvgProfiles (for 10 GPD each) + 1 Legend
#
# gpds = np.arange(24,44,2)
# bc = "ibb1"
#
# color_list = list(matplotlib.colors.TABLEAU_COLORS.keys())
# colormap = plt.cm.coolwarm#viridis # cmaps: viridis, plasma, inferno, magma, cividis, (tab10)
# color_index_list = np.linspace(0,1,10)
# x_tick_list = np.arange(-3,3+1,1)
# alpha_value = 0.7
#
# #paths_dict[bc+"_"+co+"_GPD"+str(gpd)]  # contains full path
# with matplotlib.rc_context({'lines.linewidth': 0.5,'font.size': 5}):
#     for co in cos:
#         fig, axs = plt.subplots(3,2, sharex=True, figsize=(3.4876, 3.4876))  # 5 AvgProfiles + 1 Legend in last
#
#         if co == "reg":
#             co_label = "REG"
#         elif co == "kbc":
#             co_label = "KBC"
#         legend_elements=[]
#         color_index = 0
#         for gpd in gpds:
#             # LOAD DATA FROM SIM
#             avg_u1 = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_1_t-avg.npy")
#             avg_u2 = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_2_t-avg.npy")
#             avg_u3 = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_3_t-avg.npy")
#
#             avg_u1_x = avg_u1[0]  # u_x component over y at pos 1
#             avg_u2_x = avg_u2[0]  # u_x component over y at pos 2
#             avg_u3_x = avg_u3[0]  # u_x component over y at pos 3
#
#             avg_u1_y = avg_u1[1]  # u_y component over y at pos 1
#             avg_u2_y = avg_u2[1]  # u_y component over y at pos 2
#             avg_u3_y = avg_u3[1]  # u_y component over y at pos 3
#
#             y_in_D = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_YinD.npy")
#
#             u1_diff_sq_mean_x = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_1_ReStress_x.npy") # contains y_in_D (index 0) and data (index 1)
#             u2_diff_sq_mean_x = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_2_ReStress_x.npy")
#             u3_diff_sq_mean_x = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_3_ReStress_x.npy")
#             u1_diff_sq_mean_y = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_1_ReStress_y.npy")
#             u2_diff_sq_mean_y = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_2_ReStress_y.npy")
#             u3_diff_sq_mean_y = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_3_ReStress_y.npy")
#
#             u1_diff_xy_mean = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_1_ReShearStress.npy")
#             u2_diff_xy_mean = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_2_ReShearStress.npy")
#             u3_diff_xy_mean = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_3_ReShearStress.npy")
#
#             # PLOT DATA IN FIGURE
#             axs[0,0].plot(y_in_D, avg_u1_x, y_in_D, avg_u2_x - 1, y_in_D, avg_u3_x - 2, color=colormap(color_index_list[color_index]))
#             axs[0,1].plot(y_in_D, avg_u1_y, y_in_D, avg_u2_y - 1, y_in_D, avg_u3_y - 2, color=colormap(color_index_list[color_index]))
#             axs[1,0].plot(u1_diff_sq_mean_x[0],u1_diff_sq_mean_x[1], u2_diff_sq_mean_x[0], u2_diff_sq_mean_x[1]-0.5, u3_diff_sq_mean_x[0],u3_diff_sq_mean_x[1]-1, color=colormap(color_index_list[color_index]))
#             axs[1,1].plot(u1_diff_sq_mean_y[0],u1_diff_sq_mean_y[1], u2_diff_sq_mean_y[0], u2_diff_sq_mean_y[1]-0.5, u3_diff_sq_mean_y[0],u3_diff_sq_mean_y[1]-1, color=colormap(color_index_list[color_index]))
#             axs[2,0].plot(u1_diff_xy_mean[0], u1_diff_xy_mean[1], u2_diff_xy_mean[0], u2_diff_xy_mean[1]-0.5, u3_diff_xy_mean[0], u3_diff_xy_mean[1]-1, color=colormap(color_index_list[color_index]))
#
#             # set invisible point for legend entry:
#             axs[2,1].plot(-1000,-1000, color=colormap(color_index_list[color_index]),
#                           #label="GPD "+str(gpd))
#                           label=str(gpd))
#             legend_elements.append(matplotlib.lines.Line2D([0],[0], color=colormap(color_index_list[color_index]), lw=1, label=str(gpd)))
#             color_index = color_index+1
#
#         for ax in axs.flat:
#             ax.set_xticks(ticks=x_tick_list)
#
#         axs[0,0].set_ylabel(r"$\bar{u}_{x}$/$u_{char}$")
#         axs[0,0].set_ylim([-2.5, +2])
#         axs[0,0].set_xlim([-3, 3])
#         axs[0,0].set_yticks(ticks=[-2,-1,0,1,2])
#         axs[0,0].set_yticks(ticks=[-2.5,-1.5,-0.5,0.5,1.5,2.5], minor=True)
#         # axs[0,0].text(-2.8, 1.3, "X/D = 1.06", fontsize=4.5)
#         # axs[0,0].text(-2.8, 0.3, "X/D = 1.54", fontsize=4.5)
#         # axs[0,0].text(-2.8, -0.7, "X/D = 1.54", fontsize=4.5)
#
#         axs[0,1].set_ylabel(r"$\bar{u}_{y}$/$u_{char}$")
#         axs[0,1].set_ylim([-2.5, +0.5])
#         axs[0,1].set_xlim([-3, 3])
#         axs[0,1].set_yticks(ticks=[-2,-1,0])
#         axs[0,1].set_yticks(ticks=[-2.5,-1.5,-0.5, 0.5], minor=True)
#
#         axs[1,0].set_ylabel(r"$\overline{u_{x}'u_{x}'}$/$u_{char}^2$")
#         axs[1,0].set_ylim([-1.2, 0.3])
#         axs[1,0].set_xlim([-3, 3])
#         axs[1,0].set_yticks(ticks=[-1,-0.5,0])
#
#         axs[1,1].set_ylabel(r"$\overline{u_{y}'u_{y}'}$/$u_{char}^2$")
#         axs[1,1].set_ylim([-1.2, 0.3])
#         axs[1,1].set_xlim([-3, 3])
#         axs[1,1].set_yticks(ticks=[-1,-0.5,0])
#
#         axs[2,0].set_xlabel("y/D")
#         axs[2,0].set_ylabel(r"$\overline{u_{x}'u_{y}'}$/$u_{char}^2$")
#         axs[2,0].set_ylim([-1.2, 0.2])
#         axs[2,0].set_xlim([-3, 3])
#         axs[2,0].set_yticks(ticks=[-1,-0.5,0])
#
#         axs[2,1].set_xlabel("y/D")
#         axs[2,1].set_ylim([0, 1])
#         axs[2,1].tick_params(left=False, labelleft=False)
#         axs[2,1].legend(handles=legend_elements,edgecolor="white",ncol=2, loc="lower left")
#         axs[2,1].text(-2.7,0.85,"Collision operator: "+co_label)
#         axs[2,1].text(-2.7,0.63,"Resolution in GPD:")
#
#         plt.savefig("../plots/3D_Re3900_AvgProfile_GPD_"+co+".png")
#
#         # ax.set_xlabel("y/D")
#         # ax.set_ylabel(r"$\bar{u}_{x}$/$u_{char}$")
#         # ax.set_ylim([-2.5, +2])
#         # ax.set_xlim([-3, 3])
#
#         # ax.set_xlabel("y/D")
#         # ax.set_ylabel(r"$\bar{u}_{y}$/$u_{char}$")
#         # ax.set_ylim([-2.5, +1.5])
#         # ax.set_xlim([-3, 3])
#
#         # ax.set_xlabel("y/D")
#         # ax.set_ylabel(r"$\overline{u_{x}'u_{x}'}$/$u_{char}^2$")
#         # ax.set_ylim([-1.2, 0.8])
#         # ax.set_xlim([-3, 3])
#
#         # ax.set_xlabel("y/D")
#         # ax.set_ylabel(r"$\overline{u_{y}'u_{y}'}$/$u_{char}^2$")
#         # ax.set_ylim([-1.2, 0.8])
#         # ax.set_xlim([-3, 3])
#
#         # ax.set_xlabel("y/D")
#         # ax.set_ylabel(r"$\overline{u_{x}'u_{y}'}$/$u_{char}^2$")
#         # ax.set_ylim([-1.2, 0.8])
#         # ax.set_xlim([-3, 3])
#
#
#
# ### highest Res for both CO against literature
# gpd=42
# with matplotlib.rc_context({'lines.linewidth': 0.5,'font.size': 5, 'lines.markersize': 1.2, 'lines.markeredgewidth': 0.3}):
#     for co in cos:
#         fig, axs = plt.subplots(3,2, sharex=True, figsize=(3.4876, 3.4876))  # 5 AvgProfiles + 1 Legend in last
#
#         if co == "reg":
#             co_label = "REG"
#         elif co == "kbc":
#             co_label = "KBC"
#
#         # LOAD DATA FROM SIM
#         avg_u1 = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_1_t-avg.npy")
#         avg_u2 = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_2_t-avg.npy")
#         avg_u3 = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_3_t-avg.npy")
#
#         avg_u1_x = avg_u1[0]  # u_x component over y at pos 1
#         avg_u2_x = avg_u2[0]  # u_x component over y at pos 2
#         avg_u3_x = avg_u3[0]  # u_x component over y at pos 3
#
#         avg_u1_y = avg_u1[1]  # u_y component over y at pos 1
#         avg_u2_y = avg_u2[1]  # u_y component over y at pos 2
#         avg_u3_y = avg_u3[1]  # u_y component over y at pos 3
#
#         y_in_D = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_YinD.npy")
#
#         u1_diff_sq_mean_x = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_1_ReStress_x.npy") # contains y_in_D (index 0) and data (index 1)
#         u2_diff_sq_mean_x = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_2_ReStress_x.npy")
#         u3_diff_sq_mean_x = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_3_ReStress_x.npy")
#         u1_diff_sq_mean_y = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_1_ReStress_y.npy")
#         u2_diff_sq_mean_y = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_2_ReStress_y.npy")
#         u3_diff_sq_mean_y = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_3_ReStress_y.npy")
#
#         u1_diff_xy_mean = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_1_ReShearStress.npy")
#         u2_diff_xy_mean = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_2_ReShearStress.npy")
#         u3_diff_xy_mean = np.load(paths_dict[bc+"_"+co+"_GPD"+str(gpd)][0]+"/AvgVelocity_Data" + "/AvgVelocity_3_ReShearStress.npy")
#
#         # PLOT DATA IN FIGURE
#         my_data = axs[0,0].plot(y_in_D, avg_u1_x, y_in_D, avg_u2_x - 1, y_in_D, avg_u3_x - 2, color="red", label="present")
#         ref_LS = axs[0,0].plot(p1_LS1993_ux[:, 0], p1_LS1993_ux[:, 1], p2_LS1993_ux[:, 0], p2_LS1993_ux[:, 1], p3_LS1993_ux[:, 0],
#                          p3_LS1993_ux[:, 1], marker="s", fillstyle='none', color="k", label="Lorenco & Shih (1993)")
#         ref_KM = axs[0,0].plot(p1_KM2000_ux[:, 0], p1_KM2000_ux[:, 1], p2_KM2000_ux[:, 0], p2_KM2000_ux[:, 1], p3_KM2000_ux[:, 0],
#                          p3_KM2000_ux[:, 1], ls="dotted", marker="", color="k", label="Kravchenko & Moin (2000)")
#         ref_WR = axs[0,0].plot(p1_WR2008_ux[:, 0], p1_WR2008_ux[:, 1], p2_WR2008_ux[:, 0], p2_WR2008_ux[:, 1], p3_WR2008_ux[:, 0],
#                          p3_WR2008_ux[:, 1], ls="dashdot", marker="", color="k", label="Wissink & Rodi (2008)")
#         ref_DI = axs[0,0].plot(p1_DI2018_ux[:, 0], p1_DI2018_ux[:, 1], p2_DI2018_ux[:, 0], p2_DI2018_ux[:, 1], p3_DI2018_ux[:, 0],
#                          p3_DI2018_ux[:, 1], ls="--", marker="", color="tab:blue", label="Di Ilio et al. (2018)")
#
#         axs[0,1].plot(y_in_D, avg_u1_y, y_in_D, avg_u2_y - 1, y_in_D, avg_u3_y - 2, color="red", label="present")
#         ref_LS = axs[0,1].plot(p1_LS1993_uy[:, 0], p1_LS1993_uy[:, 1], p2_LS1993_uy[:, 0], p2_LS1993_uy[:, 1], p3_LS1993_uy[:, 0],
#                      p3_LS1993_uy[:, 1], ls="",marker="s", fillstyle='none', color="k", label="Lorenco & Shih (1993)")
#         ref_KM = axs[0,1].plot(p1_KM2000_uy[:, 0], p1_KM2000_uy[:, 1], p2_KM2000_uy[:, 0], p2_KM2000_uy[:, 1], p3_KM2000_uy[:, 0],
#                          p3_KM2000_uy[:, 1], ls="dotted", marker="", color="k", label="Kravchenko & Moin (2000)")
#         ref_WR = axs[0,1].plot(p1_WR2008_uy[:, 0], p1_WR2008_uy[:, 1], p2_WR2008_uy[:, 0], p2_WR2008_uy[:, 1], p3_WR2008_uy[:, 0],
#                          p3_WR2008_uy[:, 1], ls="dashdot",  marker="", color="k", label="Wissink & Rodi (2008)")
#         ref_DI = axs[0,1].plot(p1_DI2018_uy[:, 0], p1_DI2018_uy[:, 1], p2_DI2018_uy[:, 0], p2_DI2018_uy[:, 1], p3_DI2018_uy[:, 0],
#                          p3_DI2018_uy[:, 1], ls="--", marker="", color="tab:blue", label="Di Ilio et al. (2018)")
#
#         axs[1,0].plot(u1_diff_sq_mean_x[0],u1_diff_sq_mean_x[1], u2_diff_sq_mean_x[0], u2_diff_sq_mean_x[1]-0.5, u3_diff_sq_mean_x[0],u3_diff_sq_mean_x[1]-1, color="red", label="present")
#         ref_LS = axs[1,0].plot(p2_LS1993_uxux[:, 0], p2_LS1993_uxux[:, 1], ls="", marker="s", fillstyle='none', color="k", label="Lorenco & Shih (1993)")
#         ref_R = axs[1,0].plot(p1_R2016_uxux[:, 0], p1_R2016_uxux[:, 1], p3_R2016_uxux[:, 0], p3_R2016_uxux[:, 1],
#                         p3_R2016_uxux[:, 0], p3_R2016_uxux[:, 1], ls="--", marker="", color="k", label="Rajani et al. (2016)")
#         ref_KM = axs[1,0].plot(p1_KM2000_uxux[:, 0], p1_KM2000_uxux[:, 1], p2_KM2000_uxux[:, 0], p2_KM2000_uxux[:, 1],
#                          p3_KM2000_uxux[:, 0], p3_KM2000_uxux[:, 1], ls="dotted", marker="", color="k", label="Kravchenko & Moin (2000)")
#         ref_BM = axs[1,0].plot(p2_BM1994_uxux[:, 0], p2_BM1994_uxux[:, 1], ls="--", marker="", color="g", label="Beaudan & Moin (1994)")
#         ref_DI = axs[1,0].plot(p1_DI2018_uxux[:, 0], p1_DI2018_uxux[:, 1], p2_DI2018_uxux[:, 0], p2_DI2018_uxux[:, 1],
#                          p3_DI2018_uxux[:, 0], p3_DI2018_uxux[:, 1], ls="--", marker="", color="tab:blue", label="Di Ilio et al. (2018)")
#
#         axs[1,1].plot(u1_diff_sq_mean_y[0],u1_diff_sq_mean_y[1], u2_diff_sq_mean_y[0], u2_diff_sq_mean_y[1]-0.5, u3_diff_sq_mean_y[0],u3_diff_sq_mean_y[1]-1, color="red", label="present")
#         ref_BM = axs[1,1].plot(p2_BM1994_uyuy[:, 0], p2_BM1994_uyuy[:, 1], ls="--", marker="", color="g", label="Beaudan & Moin (1994)")
#         ref_LS = axs[1,1].plot(p2_LS1993_uyuy[:, 0], p2_LS1993_uyuy[:, 1], ls="", marker="s", fillstyle='none', color="k", label="Lorenco & Shih (1993)")
#         ref_R = axs[1,1].plot(p1_R2016_uyuy[:, 0], p1_R2016_uyuy[:, 1], p3_R2016_uyuy[:, 0], p3_R2016_uyuy[:, 1],
#                         p3_R2016_uyuy[:, 0], p3_R2016_uyuy[:, 1], ls="--", marker="", color="k", label="Rajani et al. (2016)")
#         ref_DI = axs[1,1].plot(p1_DI2018_uyuy[:, 0], p1_DI2018_uyuy[:, 1], p2_DI2018_uyuy[:, 0], p2_DI2018_uyuy[:, 1],
#                          p3_DI2018_uyuy[:, 0], p3_DI2018_uyuy[:, 1], ls="--", marker="", color="tab:blue", label="Di Ilio et al. (2018)")
#
#         axs[2,0].plot(u1_diff_xy_mean[0], u1_diff_xy_mean[1], u2_diff_xy_mean[0], u2_diff_xy_mean[1]-0.5, u3_diff_xy_mean[0], u3_diff_xy_mean[1]-1, color="red", label="present")
#         ref_BM = axs[2,0].plot(p2_BM1994_uxuy[:, 0], p2_BM1994_uxuy[:, 1], ls="--", marker="", color="g", label="Beaudan & Moin (1994)")
#         ref_LS = axs[2,0].plot(p2_LS1993_uxuy[:, 0], p2_LS1993_uxuy[:, 1], ls="", marker="s", fillstyle='none', color="k", label="Lorenco & Shih (1993)")
#         ref_R = axs[2,0].plot(p1_R2016_uxuy[:, 0], p1_R2016_uxuy[:, 1], p3_R2016_uxuy[:, 0], p3_R2016_uxuy[:, 1],
#                         p3_R2016_uxuy[:, 0], p3_R2016_uxuy[:, 1], ls="--", marker="", color="k", label="Rajani et al. (2016)")
#         ref_DI = axs[2,0].plot(p1_DI2018_uxuy[:, 0], p1_DI2018_uxuy[:, 1], p2_DI2018_uxuy[:, 0], p2_DI2018_uxuy[:, 1],
#                          p3_DI2018_uxuy[:, 0], p3_DI2018_uxuy[:, 1], ls="--", marker="", color="tab:blue", label="Di Ilio et al. (2018)")
#
#         # set invisible point for legend entry:
#         axs[2,1].legend(handles=[my_data[0], ref_LS[0], ref_KM[0], ref_WR[0], ref_DI[0], ref_R[0], ref_BM[0]],edgecolor="white", loc="lower center", fontsize=4.5)
#         color_index = color_index+1
#
#         for ax in axs.flat:
#             ax.set_xticks(ticks=x_tick_list)
#
#         axs[0,0].set_ylabel(r"$\bar{u}_{x}$/$u_{char}$")
#         axs[0,0].set_ylim([-2.5, +2])
#         axs[0,0].set_xlim([-3, 3])
#         axs[0,0].set_yticks(ticks=[-2,-1,0,1,2])
#         axs[0,0].set_yticks(ticks=[-2.5,-1.5,-0.5,0.5,1.5,2.5], minor=True)
#         # axs[0,0].text(-2.8, 1.3, "X/D = 1.06", fontsize=4.5)
#         # axs[0,0].text(-2.8, 0.3, "X/D = 1.54", fontsize=4.5)
#         # axs[0,0].text(-2.8, -0.7, "X/D = 1.54", fontsize=4.5)
#
#         axs[0,1].set_ylabel(r"$\bar{u}_{y}$/$u_{char}$")
#         axs[0,1].set_ylim([-2.5, +0.5])
#         axs[0,1].set_xlim([-3, 3])
#         axs[0,1].set_yticks(ticks=[-2,-1,0])
#         axs[0,1].set_yticks(ticks=[-2.5,-1.5,-0.5, 0.5], minor=True)
#
#         axs[1,0].set_ylabel(r"$\overline{u_{x}'u_{x}'}$/$u_{char}^2$")
#         axs[1,0].set_ylim([-1.2, 0.3])
#         axs[1,0].set_xlim([-3, 3])
#         axs[1,0].set_yticks(ticks=[-1,-0.5,0])
#
#         axs[1,1].set_ylabel(r"$\overline{u_{y}'u_{y}'}$/$u_{char}^2$")
#         axs[1,1].set_ylim([-1.2, 0.3])
#         axs[1,1].set_xlim([-3, 3])
#         axs[1,1].set_yticks(ticks=[-1,-0.5,0])
#
#         axs[2,0].set_xlabel("y/D")
#         axs[2,0].set_ylabel(r"$\overline{u_{x}'u_{y}'}$/$u_{char}^2$")
#         axs[2,0].set_ylim([-1.2, 0.2])
#         axs[2,0].set_xlim([-3, 3])
#         axs[2,0].set_yticks(ticks=[-1,-0.5,0])
#
#         axs[2,1].set_xlabel("y/D")
#         axs[2,1].tick_params(left=False, labelleft=False)
#         axs[2,1].set_ylim([0, 1])
#         #axs[2,1].text(-0.8,0.8,co_label, fontsize=11)
#         axs[2,1].text(-2.7,0.85,"Collision operator: "+co_label)
#
#         plt.savefig("../plots/3D_Re3900_AvgProfile_vsLit_"+co+".png")