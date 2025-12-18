"""
    THIS FILE CONTAINS THE UTILITIES TO PROCESS AND PLOT THE REPORTED DATA FROM
     THE CYLINDER OBSTACLE SIMULATION SCRIPT
     - plotting of force coefficients
     - analysis of periodic (force coefficient) signals
     - drawing of circular cylinder mask in 2D
     - plotting- and analysis utility for average velocity and reynolds
        stress profiles
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
import traceback
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from collections import Counter


def plot_force_coefficient(data_array: np.ndarray, ylabel: str, ylim: tuple[float, float],
                           secax_functions_tuple: tuple[Any,Any], filenamebase,
                           save_timeseries = False, periodic_start=0, adjust_ylim=False):
    """
        - plot force coefficient timeseries
        - (opt.) save png to outdir
        - contains exception-capture functionality if data is not plot-abel,
          save-able etc.
    """

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

    # PLOT with ylim automatically ADJUSTED to fit data (additionally)
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
            print(f"(WARNING!) saving of {ylabel} PLOT "
                  f"to {filenamebase}_ylimadjusted.png didn't work...")
            print("\n--- Python Stack Trace ---")
            full_trace = traceback.format_exc()
            print(full_trace)
            print("--------------------------\n")

    # SAVE .txt timeseries
    #TODO (optional): make this a seperate function...
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



def analyze_periodic_timeseries(data_array: np.ndarray, periodic_start_rel: float,prominence: float = None, name: str = "periodic timeseries", outdir=None, pu_per_step = None, verbose=False):
    """
       ANALYZE PERIODIC SIGNAL (verbose = plot peak-finding),
       - outputs: min, max, mean (simple + corrected for integer number of periods)
       - RETURN: dictionary mit: Name, min, max, mean_simple, mean_n_periodic,
                mean-min, mean-max (v.a. für Cl, nach neuen Skripten)
       - print error if peak-finding didn't work
       - do FFT, print FFT spectrum
       - detect frequency of signal by sine-functino fitting
            (can be more accurate than FFT for strictly sinusoidal signals!)
    """
    values_periodic = data_array[int(data_array.shape[0] * periodic_start_rel - 1):, 2]
    steps_LU_periodic = data_array[int(data_array.shape[0] * periodic_start_rel - 1):, 0]
    mean_periodcorrected = None
    max_mean = None # mean maximum value from all high peaks
    min_mean = None # mean minimum value from all low peaks
    if prominence is None:
        prominence = ((values_periodic.max() - values_periodic.min()) * 0.5)
        # The prominence value is up for debate;
        #  The value given here only reliably catches all peaks,
        #  if the signal is simple and periodically converged


    try:
        peaks_max = find_peaks(values_periodic, prominence=prominence)
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
        print(f"(WARNING!) peak finding for {name} didn't work... "
              f"This might just be because there is no converged periodic region,"
              f"so don't worry if you expected this!")
        if verbose:
            print("Analyze Periodic Timeseries (verbose):-> see Python Stack Trace below:")

            print("\n--- Python Stack Trace ---")
            full_trace = traceback.format_exc()
            print(full_trace)
            print("--------------------------\n")

    # simple mean, max, min calculation
    mean_simple = values_periodic.mean()
    min_simple = values_periodic.min()
    max_simple = values_periodic.max()

    # sine-fit
    # (for better statistics and frequency analysis of purely sinusoidal signals)
    def sine_func(xx, a, b, c, d):
        return a * np.sin(2 * np.pi * b * xx + c) + d

    frequency_fit = None
    try:
        coefficients, values = curve_fit(sine_func, steps_LU_periodic, values_periodic,
                                         p0=(0.7, 0.2, 0.5, 0))
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
        if verbose:
            print("\n--- Python Stack Trace ---")
            full_trace = traceback.format_exc()
            print(full_trace)
            print("--------------------------\n")

    # FFT
    #  run FFT on signal and get dominant frequency
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
                # plot spectrum |X|(f)
                plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
                plt.xlabel("Freq (Hz)")
                plt.ylabel("FFT Amplitude |X(freq)|")
                plt.xlim(0, 1)
                # print("max. Amplitude np.abx(X).max():", np.abs(X).max())   # uncomment for debugging

                # ylim, where highes peak is on left half of full spectrum:
                plt.ylim(0, np.abs(X[:int(X.shape[0] * 0.5)]).max())

            if outdir is not None:
                plt.savefig(outdir + f"/{name}_fft.png")

            freq_res = freq[1] - freq[0]  # frequency-resolution

            # get |X| Amplitude for left half of full spectrum
            X_abs = np.abs(X[:int(X.shape[0] * 0.4)])
            freq_peak = freq[np.argmax(X_abs)]  # find frequency with the highest amplitude
        except Exception as e:
            print(f"(WARNING!) fft for {name} didn't work...")
            if verbose:
                print("\n--- Python Stack Trace ---")
                full_trace = traceback.format_exc()
                print(full_trace)
                print("--------------------------\n")


    return {"mean_simple": mean_simple, "mean_periodcorrected": mean_periodcorrected,
            "min_simple": min_simple, "max_simple": max_simple, "max_mean": max_mean,
            "min_mean": min_mean, "frequency_fit": frequency_fit, "frequency_fft": freq_peak,
            "fft_resolution": freq_res}


def draw_circular_mask(flow, gridpoints_per_diameter, output_data=False,
                       filebase=".", print_data=False):
    """
        calculate and draw a 2D representation of:
        - the circular cylinder
        - the basic solid mask
    """

    grid_x = gridpoints_per_diameter + 2

    if print_data:
        print("GPD = " + str(gridpoints_per_diameter))
    # define radius and position for a symmetrical circular Cylinder-Obstacle
    radius_lu = 0.5 * gridpoints_per_diameter
    y_pos_lu = 0.5 * grid_x + 0.5
    x_pos_lu = y_pos_lu

    # get x,y,z meshgrid of the domain (LU)

    # tupel of list indizes (1-n (non-zero-based!))
    xyz = tuple(np.linspace(1, n, n) for n in (grid_x, grid_x))
    # meshgrid of x- and y- indizes -> * unpacks the tuple to be two values and now a tuple
    x_lu, y_lu = np.meshgrid(*xyz, indexing='ij')

    # define cylinder (LU) (circle)
    obstacle_mask_for_visualization = np.sqrt(
        (x_lu - x_pos_lu) ** 2 + (y_lu - y_pos_lu) ** 2) < radius_lu

    nx, ny = obstacle_mask_for_visualization.shape  # number of x- and y-nodes (Skalar)

    # for all the solid nodes, neighboring fluid nodes
    rand_mask = np.zeros((nx, ny), dtype=bool)

    # same, but including q-dimension
    rand_mask_f = np.zeros((flow.stencil.q, nx, ny), dtype=bool)

    rand_xq = []  # list of all x-values (incl. q-multiplicity)
    rand_yq = []  # list of all y-values (incl. q-multiplicity)

    # np.array: list of
    #   (a) x-coordinates and
    #   (b) y-coordinates of the obstacle_mask_for_visualization
    a, b = np.where(obstacle_mask_for_visualization)
    # ...to iterate over all boundary/object/wall nodes
    for p in range(0,len(a)):
        # for all True-nodes in obstacle_mask_for_visualization
        for i in range(0,flow.stencil.q):
            # for all stencil directions c_i (lattice.stencil.e)
            try:
                # try in case the neighboring cell does not exist
                # (an f pointing out of the simulation domain)
                if not obstacle_mask_for_visualization[
                    a[p] + flow.stencil.e[i][0], b[p] + flow.stencil.e[i][1]]:
                    # if neighbor in +(e_x, e_y; e is c_i) is False,
                    # we are on the object-surface (self True with neighbor False)
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

    # list of all radii (without q-dimension) in LU
    radii = np.zeros_like(rand_x, dtype=float)
    for p in range(0, len(rand_x)):  # for all nodes
        # calculate distance to circle center
        radii[p] = np.sqrt((rand_x[p] - x_pos) ** 2 + (rand_y[p] - y_pos) ** 2)
        if radii[p] > r_max:
            r_max = radii[p]
        if radii[p] < r_min:
            r_min = radii[p]

    # calculate all radii (with q-multiplicity)
    radii_q = np.zeros_like(rand_xq, dtype=float)
    for p in range(0, len(rand_xq)):
        radii_q[p] = np.sqrt((rand_xq[p] - x_pos) ** 2 + (rand_yq[p] - y_pos) ** 2)

    ### all relative radii in relation to gpd/2
    radii_relative = radii / (radius_lu - 0.5)
        # (subtract 0.5 because "true" boundary location is 0.5LU
        #  further out than node-coordinates)
    radii_q_relative = radii_q / (radius_lu - 0.5)

    # calc. mean rel_radius
    r_rel_mean = sum(radii_relative) / len(radii_relative)
    rq_rel_mean = sum(radii_q_relative) / len(radii_q_relative)

    ## AREA calculation
    area_theory = np.pi * (gridpoints_per_diameter / 2) ** 2  # area = pi*r² in LU²
    # area in LU = number of nodes, because every node has a cell of 1LU x 1LU around it
    area = len(a)

    if output_data:
        output_file = open(filebase + "/obstacle_mask_info.txt", "a")
        output_file.write("GPD = " + str(gridpoints_per_diameter) + "\n")
        output_file.write(
            "\nr_rel_mean: " + str(sum(radii_relative) / len(radii_relative)))
        output_file.write("\nrq_rel_mean: " + str(
            sum(radii_q_relative) / len(radii_q_relative)))
        output_file.write("\nr_rel_min: " + str(r_max / (radius_lu - 0.5)))
        output_file.write("\nr_rel_max: " + str(r_min / (radius_lu - 0.5)))
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

    # grid thickness, circle, node marker
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


class ProfilePlotter:
    """
        utility to handle profile reporter data
        - load reference values
        - process profile reporter data
        - plot and save data with or without references
    """


    def __init__(self, flow, output_path, reference_data_path, i_timeseries,
                 u_timeseries1, u_timeseries2, u_timeseries3):
        self.flow = flow
        self.output_path = output_path
        self.i_timeseries = i_timeseries
        self.u_timeseries1 = u_timeseries1
        self.u_timeseries2 = u_timeseries2
        self.u_timeseries3 = u_timeseries3

        os.makedirs(self.output_path + "/ProfileReporter_Data/")
        
        self.import_profile_reference_data(reference_data_path)
    
    def import_profile_reference_data(self, data_path):
        # import reference data:
        # (data is: first column Y/D, second column u_d/u_char)

        # ux
        self.p1_LS1993_ux = np.genfromtxt(
            data_path + 'Fig09_ux_profile_pos1_LS1993.csv', delimiter=';')
        self.p2_LS1993_ux = np.genfromtxt(
            data_path + 'Fig09_ux_profile_pos2_LS1993.csv', delimiter=';')
        self.p3_LS1993_ux = np.genfromtxt(
            data_path + 'Fig09_ux_profile_pos3_LS1993.csv', delimiter=';')

        self.p1_KM2000_ux = np.genfromtxt(
            data_path + 'Fig09_ux_profile_pos1_KM2000.csv', delimiter=';')
        self.p2_KM2000_ux = np.genfromtxt(
            data_path + 'Fig09_ux_profile_pos2_KM2000.csv', delimiter=';')
        self.p3_KM2000_ux = np.genfromtxt(
            data_path + 'Fig09_ux_profile_pos3_KM2000.csv', delimiter=';')

        self.p1_WR2008_ux = np.genfromtxt(
            data_path + 'Fig09_ux_profile_pos1_WR2008.csv', delimiter=';')
        self.p2_WR2008_ux = np.genfromtxt(
            data_path + 'Fig09_ux_profile_pos2_WR2008.csv', delimiter=';')
        self.p3_WR2008_ux = np.genfromtxt(
            data_path + 'Fig09_ux_profile_pos3_WR2008.csv', delimiter=';')

        self.p1_DI2018_ux = np.genfromtxt(
            data_path + 'Fig09_ux_profile_pos1_DI2018.csv', delimiter=';')
        self.p2_DI2018_ux = np.genfromtxt(
            data_path + 'Fig09_ux_profile_pos2_DI2018.csv', delimiter=';')
        self.p3_DI2018_ux = np.genfromtxt(
            data_path + 'Fig09_ux_profile_pos3_DI2018.csv', delimiter=';')

        # uy
        self.p1_LS1993_uy = np.genfromtxt(
            data_path + 'Fig10_uy_profile_pos1_LS1993.csv', delimiter=';')
        self.p2_LS1993_uy = np.genfromtxt(
            data_path + 'Fig10_uy_profile_pos2_LS1993.csv', delimiter=';')
        self.p3_LS1993_uy = np.genfromtxt(
            data_path + 'Fig10_uy_profile_pos3_LS1993.csv', delimiter=';')

        self.p1_KM2000_uy = np.genfromtxt(
            data_path + 'Fig10_uy_profile_pos1_KM2000.csv', delimiter=';')
        self.p2_KM2000_uy = np.genfromtxt(
            data_path + 'Fig10_uy_profile_pos2_KM2000.csv', delimiter=';')
        self.p3_KM2000_uy = np.genfromtxt(
            data_path + 'Fig10_uy_profile_pos3_KM2000.csv', delimiter=';')

        self.p1_WR2008_uy = np.genfromtxt(
            data_path + 'Fig10_uy_profile_pos1_WR2008.csv', delimiter=';')
        self.p2_WR2008_uy = np.genfromtxt(
            data_path + 'Fig10_uy_profile_pos2_WR2008.csv', delimiter=';')
        self.p3_WR2008_uy = np.genfromtxt(
            data_path + 'Fig10_uy_profile_pos3_WR2008.csv', delimiter=';')

        self.p1_DI2018_uy = np.genfromtxt(
            data_path + 'Fig10_uy_profile_pos1_DI2018.csv', delimiter=';')
        self.p2_DI2018_uy = np.genfromtxt(
            data_path + 'Fig10_uy_profile_pos2_DI2018.csv', delimiter=';')
        self.p3_DI2018_uy = np.genfromtxt(
            data_path + 'Fig10_uy_profile_pos3_DI2018.csv', delimiter=';')

        # uxux
        self.p1_DI2018_uxux = np.genfromtxt(
            data_path + 'Fig11_uxux_profile_pos1_DI2018.csv', delimiter=';')
        self.p1_KM2000_uxux = np.genfromtxt(
            data_path + 'Fig11_uxux_profile_pos1_KM2000.csv', delimiter=';')
        self.p1_R2016_uxux = np.genfromtxt(
            data_path + 'Fig11_uxux_profile_pos1_R2016.csv', delimiter=';')
        self.p2_BM1994_uxux = np.genfromtxt(
            data_path + 'Fig11_uxux_profile_pos2_BM1994.csv', delimiter=';')
        self.p2_DI2018_uxux = np.genfromtxt(
            data_path + 'Fig11_uxux_profile_pos2_DI2018.csv', delimiter=';')
        self.p2_KM2000_uxux = np.genfromtxt(
            data_path + 'Fig11_uxux_profile_pos2_KM2000.csv', delimiter=';')
        self.p2_LS1993_uxux = np.genfromtxt(
            data_path + 'Fig11_uxux_profile_pos2_LS1993.csv', delimiter=';')
        self.p2_R2016_uxux = np.genfromtxt(
            data_path + 'Fig11_uxux_profile_pos2_R2016.csv', delimiter=';')
        self.p3_DI2018_uxux = np.genfromtxt(
            data_path + 'Fig11_uxux_profile_pos3_DI2018.csv', delimiter=';')
        self.p3_KM2000_uxux = np.genfromtxt(
            data_path + 'Fig11_uxux_profile_pos3_KM2000.csv', delimiter=';')
        self.p3_R2016_uxux = np.genfromtxt(
            data_path + 'Fig11_uxux_profile_pos3_R2016.csv', delimiter=';')

        # uyuy
        self.p1_DI2018_uyuy = np.genfromtxt(
            data_path + 'Fig12_uyuy_profile_pos1_DI2018.csv', delimiter=';')
        self.p1_R2016_uyuy = np.genfromtxt(
            data_path + 'Fig12_uyuy_profile_pos1_R2016.csv', delimiter=';')
        self.p2_BM1994_uyuy = np.genfromtxt(
            data_path + 'Fig12_uyuy_profile_pos2_BM1994.csv', delimiter=';')
        self.p2_DI2018_uyuy = np.genfromtxt(
            data_path + 'Fig12_uyuy_profile_pos2_DI2018.csv', delimiter=';')
        self.p2_LS1993_uyuy = np.genfromtxt(
            data_path + 'Fig12_uyuy_profile_pos2_LS1993.csv', delimiter=';')
        self.p2_R2016_uyuy = np.genfromtxt(
            data_path + 'Fig12_uyuy_profile_pos2_R2016.csv', delimiter=';')
        self.p3_DI2018_uyuy = np.genfromtxt(
            data_path + 'Fig12_uyuy_profile_pos3_DI2018.csv', delimiter=';')
        self.p3_R2016_uyuy = np.genfromtxt(
            data_path + 'Fig12_uyuy_profile_pos3_R2016.csv', delimiter=';')

        # uxuy
        self.p1_BM1994_uxuy = np.genfromtxt(
            data_path + 'Fig13_uxuy_profile_pos1_BM1994.csv', delimiter=';')
        self.p1_DI2018_uxuy = np.genfromtxt(
            data_path + 'Fig13_uxuy_profile_pos1_DI2018.csv', delimiter=';')
        self.p1_R2016_uxuy = np.genfromtxt(
            data_path + 'Fig13_uxuy_profile_pos1_R2016.csv', delimiter=';')
        self.p2_BM1994_uxuy = np.genfromtxt(
            data_path + 'Fig13_uxuy_profile_pos2_BM1994.csv', delimiter=';')
        self.p2_DI2018_uxuy = np.genfromtxt(
            data_path + 'Fig13_uxuy_profile_pos2_DI2018.csv', delimiter=';')
        self.p2_LS1993_uxuy = np.genfromtxt(
            data_path + 'Fig13_uxuy_profile_pos2_LS1993.csv', delimiter=';')
        self.p2_R2016_uxuy = np.genfromtxt(
            data_path + 'Fig13_uxuy_profile_pos2_R2016.csv', delimiter=';')
        self.p3_BM1994_uxuy = np.genfromtxt(
            data_path + 'Fig13_uxuy_profile_pos3_BM1994.csv', delimiter=';')
        self.p3_DI2018_uxuy = np.genfromtxt(
            data_path + 'Fig13_uxuy_profile_pos3_DI2018.csv', delimiter=';')
        self.p3_R2016_uxuy = np.genfromtxt(
            data_path + 'Fig13_uxuy_profile_pos3_R2016.csv', delimiter=';')

    def process_data(self, save=False):
        """CALCULATE temporal velocity averages"""
        avg_u1_temporal = np.mean(np.array(self.u_timeseries1), axis=0)
        avg_u2_temporal = np.mean(np.array(self.u_timeseries2), axis=0)
        avg_u3_temporal = np.mean(np.array(self.u_timeseries3), axis=0)

        self.avg_u1_x = avg_u1_temporal[0]
        self.avg_u1_y = avg_u1_temporal[1]

        self.avg_u2_x = avg_u2_temporal[0]
        self.avg_u2_y = avg_u2_temporal[1]

        self.avg_u3_x = avg_u3_temporal[0]
        self.avg_u3_y = avg_u3_temporal[1]
        
        if save:
            np.save(
                self.output_path + "/ProfileReporter_Data" + "/ProfileReporter_"
                + "pos1" + "_t-avg.npy", avg_u1_temporal)
            np.save(
                self.output_path + "/ProfileReporter_Data" + "/ProfileReporter_"
                + "pos2" + "_t-avg.npy", avg_u2_temporal)
            np.save(
                self.output_path + "/ProfileReporter_Data" + "/ProfileReporter_"
                + "pos3" + "_t-avg.npy", avg_u3_temporal)

        # Y_inD for y-axis and plotting
        self.y_in_D = ((np.arange(self.avg_u1_x.shape[0]) + 1 - self.flow.y_pos_lu)
                       / self.flow.char_length_lu)
        if save:
            np.save(
                self.output_path + "/ProfileReporter_Data" + "/ProfileReporter_YinD.npy",
                self.y_in_D)

        # CALCULATE turbulent reynolds stresses
        # diff between timeseries and time_average -> u'
        u1_diff = self.u_timeseries1 - avg_u1_temporal
        u2_diff = self.u_timeseries2 - avg_u2_temporal
        u3_diff = self.u_timeseries3 - avg_u3_temporal

        # square of diff -> u'^2
        u1_diff_sq = u1_diff ** 2
        u2_diff_sq = u2_diff ** 2
        u3_diff_sq = u3_diff ** 2

        # ux'*uy'
        u1_diff_xy = u1_diff[:, 0, :] * u1_diff[:, 1, :]
        u2_diff_xy = u2_diff[:, 0, :] * u2_diff[:, 1, :]
        u3_diff_xy = u3_diff[:, 0, :] * u3_diff[:, 1, :]

        # time_average of u'² and ux'uy'
        self.u1_diff_sq_mean = np.mean(u1_diff_sq, axis=0)  # time average
        self.u2_diff_sq_mean = np.mean(u2_diff_sq, axis=0)  # time average
        self.u3_diff_sq_mean = np.mean(u3_diff_sq, axis=0)  # time average
        self.u1_diff_xy_mean = np.mean(u1_diff_xy, axis=0)  # time average
        self.u2_diff_xy_mean = np.mean(u2_diff_xy, axis=0)  # time average
        self.u3_diff_xy_mean = np.mean(u3_diff_xy, axis=0)  # time average

        if save:  # save reynolds stresses
            np.save(
                self.output_path + "/ProfileReporter_Data" + "/ProfileReporter_1_ReStress_x.npy",
                np.array([self.y_in_D, self.u1_diff_sq_mean[0]]))
            np.save(
                self.output_path + "/ProfileReporter_Data" + "/ProfileReporter_2_ReStress_x.npy",
                np.array([self.y_in_D, self.u2_diff_sq_mean[0]]))
            np.save(
                self.output_path + "/ProfileReporter_Data" + "/ProfileReporter_3_ReStress_x.npy",
                np.array([self.y_in_D, self.u3_diff_sq_mean[0]]))
            np.save(
                self.output_path + "/ProfileReporter_Data" + "/ProfileReporter_1_ReStress_y.npy",
                np.array([self.y_in_D, self.u1_diff_sq_mean[1]]))
            np.save(
                self.output_path + "/ProfileReporter_Data" + "/ProfileReporter_2_ReStress_y.npy",
                np.array([self.y_in_D, self.u2_diff_sq_mean[1]]))
            np.save(
                self.output_path + "/ProfileReporter_Data" + "/ProfileReporter_3_ReStress_y.npy",
                np.array([self.y_in_D, self.u3_diff_sq_mean[1]]))
            np.save(
                self.output_path + "/ProfileReporter_Data" + "/ProfileReporter_1_ReShearStress.npy",
                np.array([self.y_in_D, self.u1_diff_xy_mean]))
            np.save(
                self.output_path + "/ProfileReporter_Data" + "/ProfileReporter_2_ReShearStress.npy",
                np.array([self.y_in_D, self.u2_diff_xy_mean]))
            np.save(
                self.output_path + "/ProfileReporter_Data" + "/ProfileReporter_3_ReShearStress.npy",
                np.array([self.y_in_D, self.u3_diff_xy_mean]))

    def save_timeseries_to_files(self, basepath):
        """save the FULL timeseries to files"""
        # timeseries (i)    
        np.save(basepath + "/ProfileReporter_Data" + "/ProfileReporter_"
                + "_timeseries_steps.npy", np.array(self.i_timeseries))
        #timeseries (u)
        np.save(basepath + "/ProfileReporter_Data" + "/ProfileReporter_"
                + "pos1" + "_timeseries_data.npy", np.array(self.u_timeseries1))
        np.save(basepath + "/ProfileReporter_Data" + "/ProfileReporter_"
                + "pos2" + "_timeseries_data.npy", np.array(self.u_timeseries2))
        np.save(basepath + "/ProfileReporter_Data" + "/ProfileReporter_"
                + "pos3" + "_timeseries_data.npy", np.array(self.u_timeseries3))

    def plot_velocity_profiles(self, show_reference = False, save = False, show = False):
        """plot tht average velocity profiles"""
        cm = 1 / 2.54
        
        if not show_reference:
            # PLOT ux
            fig, (ax_ux, ax_uy) = plt.subplots(1, 2, constrained_layout=True,
                                               figsize=(30 * cm, 10 * cm))
            ax_ux.plot(self.y_in_D, self.avg_u1_x,
                       self.y_in_D, self.avg_u2_x,
                       self.y_in_D, self.avg_u3_x)
            ax_ux.set_xlabel("y/D")
            ax_ux.set_ylabel(r"$\bar{u}_{x}$/$u_{char}$")
            ax_ux.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])
    
            # PLOT uy
            ax_uy.plot(self.y_in_D, self.avg_u1_y,
                       self.y_in_D, self.avg_u2_y,
                       self.y_in_D, self.avg_u3_y)
            ax_uy.set_xlabel("y/D")
            ax_uy.set_ylabel(r"$\bar{u}_{y}$/$u_{char}$")
            ax_uy.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])
            
            if save:
                plt.savefig(self.output_path + "/ProfileReporter_Data"
                            + "/ProfileReporter_velocity_noReference.png")
            if show:
                plt.show()
            else:
                plt.close()

        else:
            # PLOT ux against references
            fig, ax = plt.subplots(constrained_layout=True)
            my_data = ax.plot(self.y_in_D, self.avg_u1_x,
                              self.y_in_D, self.avg_u2_x - 1,
                              self.y_in_D, self.avg_u3_x - 2)
            plt.setp(my_data, ls="-", lw=1, marker="", color="red", label="lettuce")
            ref_LS = ax.plot(self.p1_LS1993_ux[:, 0], self.p1_LS1993_ux[:, 1],
                             self.p2_LS1993_ux[:, 0], self.p2_LS1993_ux[:, 1],
                             self.p3_LS1993_ux[:, 0],
                             self.p3_LS1993_ux[:, 1])
            plt.setp(ref_LS, ls="", lw=1, marker="s", fillstyle='none',
                     color="k", label="Lorenco & Shih (1993)")
            ref_KM = ax.plot(self.p1_KM2000_ux[:, 0], self.p1_KM2000_ux[:, 1],
                             self.p2_KM2000_ux[:, 0], self.p2_KM2000_ux[:, 1],
                             self.p3_KM2000_ux[:, 0],
                             self.p3_KM2000_ux[:, 1])
            plt.setp(ref_KM, ls="dotted", lw=1.5, marker="", color="k",
                     label="Kravchenko & Moin (2000)")
            ref_WR = ax.plot(self.p1_WR2008_ux[:, 0], self.p1_WR2008_ux[:, 1],
                             self.p2_WR2008_ux[:, 0], self.p2_WR2008_ux[:, 1],
                             self.p3_WR2008_ux[:, 0],
                             self.p3_WR2008_ux[:, 1])
            plt.setp(ref_WR, ls="dashdot", lw=1.5, marker="", color="k",
                     label="Wissink & Rodi (2008)")
            ref_DI = ax.plot(self.p1_DI2018_ux[:, 0], self.p1_DI2018_ux[:, 1],
                             self.p2_DI2018_ux[:, 0], self.p2_DI2018_ux[:, 1],
                             self.p3_DI2018_ux[:, 0],
                             self.p3_DI2018_ux[:, 1])
            plt.setp(ref_DI, ls="--", lw=1.5, marker="", color="tab:blue",
                     label="Di Ilio et al. (2018)")
            ax.set_xlabel("y/D")
            ax.set_ylabel(r"$\bar{u}_{x}$/$u_{char}$")
            ax.set_ylim((-2.5, +2))
            ax.set_xlim((-3, 3))
            ax.legend(handles=[my_data[0], ref_LS[0], ref_KM[0], ref_WR[0],
                               ref_DI[0]], loc='best')
            if save:
                plt.savefig(self.output_path + "/ProfileReporter_Data"
                            + "/ProfileReporter_ux_withReference.png")
            if show:
                plt.show()
            else:
                plt.close()

            # PLOT uy against references
            fig, ax = plt.subplots(constrained_layout=True)
            my_data = ax.plot(self.y_in_D, self.avg_u1_y,
                              self.y_in_D, self.avg_u2_y - 1,
                              self.y_in_D, self.avg_u3_y - 2)
            plt.setp(my_data, ls="-", lw=1, marker="", color="red",
                     label="lettuce")
            ref_LS = ax.plot(self.p1_LS1993_uy[:, 0], self.p1_LS1993_uy[:, 1],
                             self.p2_LS1993_uy[:, 0], self.p2_LS1993_uy[:, 1],
                             self.p3_LS1993_uy[:, 0],
                             self.p3_LS1993_uy[:, 1])
            plt.setp(ref_LS, ls="", lw=1, marker="s", fillstyle='none',
                     color="k", label="Lorenco & Shih (1993)")
            ref_KM = ax.plot(self.p1_KM2000_uy[:, 0], self.p1_KM2000_uy[:, 1],
                             self.p2_KM2000_uy[:, 0], self.p2_KM2000_uy[:, 1],
                             self.p3_KM2000_uy[:, 0],
                             self.p3_KM2000_uy[:, 1])
            plt.setp(ref_KM, ls="dotted", lw=1.5, marker="", color="k",
                     label="Kravchenko & Moin (2000)")
            ref_WR = ax.plot(self.p1_WR2008_uy[:, 0], self.p1_WR2008_uy[:, 1],
                             self.p2_WR2008_uy[:, 0], self.p2_WR2008_uy[:, 1],
                             self.p3_WR2008_uy[:, 0],
                             self.p3_WR2008_uy[:, 1])
            plt.setp(ref_WR, ls="dashdot", lw=1.5, marker="", color="k",
                     label="Wissink & Rodi (2008)")
            ref_DI = ax.plot(self.p1_DI2018_uy[:, 0], self.p1_DI2018_uy[:, 1],
                             self.p2_DI2018_uy[:, 0], self.p2_DI2018_uy[:, 1],
                             self.p3_DI2018_uy[:, 0],
                             self.p3_DI2018_uy[:, 1])
            plt.setp(ref_DI, ls="--", lw=1.5, marker="", color="tab:blue",
                     label="Di Ilio et al. (2018)")
            ax.set_xlabel("y/D")
            ax.set_ylabel(r"$\bar{u}_{y}$/$u_{char}$")
            ax.set_ylim((-2.5, +1.5))
            ax.set_xlim((-3, 3))
            ax.legend(handles=[my_data[0], ref_LS[0], ref_KM[0], ref_WR[0],
                               ref_DI[0]], loc='best')
            if save:
                plt.savefig(self.output_path + "/ProfileReporter_Data"
                            + "/ProfileReporter_uy_withReference.png")
            if show:
                plt.show()
            else:
                plt.close()
        
    def plot_reynolds_stress_profiles(self, show_reference=False, save=False, show = False):
        """plot average reynolds stress profiles"""
        cm = 1 / 2.54
        if not show_reference:
            fig, (ax_xx, ax_yy, ax_xy) = plt.subplots(1, 3,
                                                      figsize=(40 * cm, 10 * cm),
                                                      constrained_layout=True)
            ax_xx.plot(self.y_in_D, self.u1_diff_sq_mean[0],
                       self.y_in_D, self.u2_diff_sq_mean[0],
                       self.y_in_D, self.u3_diff_sq_mean[0])
            ax_xx.set_xlabel("y/D")
            ax_xx.set_ylabel(r"$\overline{u_{x}'u_{x}'}$/$u_{char}^2$")
            ax_xx.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])
    
            ax_yy.plot(self.y_in_D, self.u1_diff_sq_mean[1],
                       self.y_in_D, self.u2_diff_sq_mean[1],
                       self.y_in_D, self.u3_diff_sq_mean[1])
            ax_yy.set_xlabel("y/D")
            ax_yy.set_ylabel(r"$\overline{u_{y}'u_{y}'}$/$u_{char}^2$")
            ax_yy.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])
    
            ax_xy.plot(self.y_in_D, self.u1_diff_xy_mean,
                       self.y_in_D, self.u2_diff_xy_mean,
                       self.y_in_D, self.u3_diff_xy_mean)
            ax_xy.set_xlabel("y/D")
            ax_xy.set_ylabel(r"$\overline{u_{x}'u_{y}'}$/$u_{char}^2$")
            ax_xy.legend(["x/D = 1.06", "x/D = 1.54", "x/D = 2.02"])
    
            if save:
                plt.savefig(self.output_path + "/ProfileReporter_Data"
                            + "/ProfileReporter_reynoldsStresses_noReference.png")
            if show:
                plt.show()
            else:
                plt.close()
        else:
            # plot reynolds stresses against reference
            # uxux - streamwise
            fig, ax = plt.subplots(constrained_layout=True)
            my_data = ax.plot(self.y_in_D, self.u1_diff_sq_mean[0],
                              self.y_in_D, self.u2_diff_sq_mean[0] - 0.5,
                              self.y_in_D, self.u3_diff_sq_mean[0] - 1)
            plt.setp(my_data, ls="-", lw=1, marker="", color="red",
                     label="lettuce")
            ref_LS = ax.plot(self.p2_LS1993_uxux[:, 0], self.p2_LS1993_uxux[:, 1])
            plt.setp(ref_LS, ls="", lw=1, marker="s", fillstyle='none',
                     color="k", label="Lorenco & Shih (1993)")
            ref_R = ax.plot(self.p1_R2016_uxux[:, 0], self.p1_R2016_uxux[:, 1],
                            self.p3_R2016_uxux[:, 0], self.p3_R2016_uxux[:, 1],
                            self.p3_R2016_uxux[:, 0], self.p3_R2016_uxux[:, 1])
            plt.setp(ref_R, ls="--", lw=1.5, marker="", color="k",
                     label="Rajani et al. (2016)")
            ref_KM = ax.plot(self.p1_KM2000_uxux[:, 0], self.p1_KM2000_uxux[:, 1],
                             self.p2_KM2000_uxux[:, 0], self.p2_KM2000_uxux[:, 1],
                             self.p3_KM2000_uxux[:, 0], self.p3_KM2000_uxux[:, 1])
            plt.setp(ref_KM, ls="dotted", lw=1.5, marker="", color="k",
                     label="Kravchenko & Moin (2000)")
            ref_BM = ax.plot(self.p2_BM1994_uxux[:, 0], self.p2_BM1994_uxux[:, 1])
            plt.setp(ref_BM, ls="dashdot", lw=1.5, marker="", color="k",
                     label="Beaudan & Moin (1994)")
            ref_DI = ax.plot(self.p1_DI2018_uxux[:, 0], self.p1_DI2018_uxux[:, 1],
                             self.p2_DI2018_uxux[:, 0], self.p2_DI2018_uxux[:, 1],
                             self.p3_DI2018_uxux[:, 0], self.p3_DI2018_uxux[:, 1])
            plt.setp(ref_DI, ls="--", lw=1.5, marker="", color="tab:blue",
                     label="Di Ilio et al. (2018)")
            ax.set_xlabel("y/D")
            ax.set_ylabel(r"$\overline{u_{x}'u_{x}'}$/$u_{char}^2$")
            ax.set_ylim((-1.2, 0.8))
            ax.set_xlim((-3, 3))
            ax.legend(
                handles=[my_data[0], ref_LS[0], ref_R[0], ref_KM[0], ref_BM[0],
                         ref_DI[0]], loc='best')

            if save:
                plt.savefig(self.output_path + "/ProfileReporter_Data"
                            + "/ProfileReporter_uxux_withReference.png")
            if show:
                plt.show()
            else:
                plt.close()

            # uyuy - cross-stream
            fig, ax = plt.subplots(constrained_layout=True)
            my_data = ax.plot(self.y_in_D, self.u1_diff_sq_mean[1],
                              self.y_in_D, self.u2_diff_sq_mean[1] - 0.5,
                              self.y_in_D, self.u3_diff_sq_mean[1] - 1)
            plt.setp(my_data, ls="-", lw=1, marker="", color="red",
                     label="lettuce")
            ref_BM = ax.plot(self.p2_BM1994_uyuy[:, 0], self.p2_BM1994_uyuy[:, 1])
            plt.setp(ref_BM, ls="dashdot", lw=1.5, marker="", color="k",
                     label="Beaudan & Moin (1994)")
            ref_LS = ax.plot(self.p2_LS1993_uyuy[:, 0], self.p2_LS1993_uyuy[:, 1])
            plt.setp(ref_LS, ls="", lw=1, marker="s", fillstyle='none',
                     color="k", label="Lorenco & Shih (1993)")
            ref_R = ax.plot(self.p1_R2016_uyuy[:, 0], self.p1_R2016_uyuy[:, 1],
                            self.p3_R2016_uyuy[:, 0], self.p3_R2016_uyuy[:, 1],
                            self.p3_R2016_uyuy[:, 0], self.p3_R2016_uyuy[:, 1])
            plt.setp(ref_R, ls="--", lw=1.5, marker="", color="k",
                     label="Rajani et al. (2016)")
            ref_DI = ax.plot(self.p1_DI2018_uyuy[:, 0], self.p1_DI2018_uyuy[:, 1],
                             self.p2_DI2018_uyuy[:, 0], self.p2_DI2018_uyuy[:, 1],
                             self.p3_DI2018_uyuy[:, 0], self.p3_DI2018_uyuy[:, 1])
            plt.setp(ref_DI, ls="--", lw=1.5, marker="", color="tab:blue",
                     label="Di Ilio et al. (2018)")
            ax.set_xlabel("y/D")
            ax.set_ylabel(r"$\overline{u_{y}'u_{y}'}$/$u_{char}^2$")
            ax.set_ylim((-1.2, 0.8))
            ax.set_xlim((-3, 3))
            ax.legend(handles=[my_data[0], ref_BM[0], ref_LS[0], ref_R[0],
                               ref_DI[0]], loc='best')

            if save:
                plt.savefig(self.output_path + "/ProfileReporter_Data"
                            + "/ProfileReporter_uyuy_withReference.png")
            if show:
                plt.show()
            else:
                plt.close()

            # uxuy - Reynolds shear stress
            fig, ax = plt.subplots(constrained_layout=True)
            my_data = ax.plot(self.y_in_D, self.u1_diff_xy_mean,
                              self.y_in_D, self.u2_diff_xy_mean - 0.5,
                              self.y_in_D, self.u3_diff_xy_mean - 1)
            plt.setp(my_data, ls="-", lw=1, marker="", color="red",
                     label="lettuce")
            ref_BM = ax.plot(self.p2_BM1994_uxuy[:, 0], self.p2_BM1994_uxuy[:, 1])
            plt.setp(ref_BM, ls="dashdot", lw=1.5, marker="", color="k",
                     label="Beaudan & Moin (1994)")
            ref_LS = ax.plot(self.p2_LS1993_uxuy[:, 0], self.p2_LS1993_uxuy[:, 1])
            plt.setp(ref_LS, ls="", lw=1, marker="s", fillstyle='none',
                     color="k", label="Lorenco & Shih (1993)")
            ref_R = ax.plot(self.p1_R2016_uxuy[:, 0], self.p1_R2016_uxuy[:, 1],
                            self.p3_R2016_uxuy[:, 0], self.p3_R2016_uxuy[:, 1],
                            self.p3_R2016_uxuy[:, 0], self.p3_R2016_uxuy[:, 1])
            plt.setp(ref_R, ls="--", lw=1.5, marker="", color="k",
                     label="Rajani et al. (2016)")
            ref_DI = ax.plot(self.p1_DI2018_uxuy[:, 0], self.p1_DI2018_uxuy[:, 1],
                             self.p2_DI2018_uxuy[:, 0], self.p2_DI2018_uxuy[:, 1],
                             self.p3_DI2018_uxuy[:, 0], self.p3_DI2018_uxuy[:, 1])
            plt.setp(ref_DI, ls="--", lw=1.5, marker="", color="tab:blue",
                     label="Di Ilio et al. (2018)")
            ax.set_xlabel("y/D")
            ax.set_ylabel(r"$\overline{u_{x}'u_{y}'}$/$u_{char}^2$")
            ax.set_ylim((-1.2, 0.8))
            ax.set_xlim((-3, 3))
            ax.legend(handles=[my_data[0], ref_BM[0], ref_LS[0], ref_R[0],
                               ref_DI[0]], loc='best')

            if save:
                plt.savefig(self.output_path + "/ProfileReporter_Data"
                            + "/ProfileReporter_uxuy_withReference.png")
            if show:
                plt.show()
            else:
                plt.close()
        
