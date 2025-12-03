# THIS FILE CONTAINS THE UTILITIES TO PROCESS AND PLOT THE REPORTED DATA FROM THE CYLINDER-SIMULATION

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable
import traceback
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from jedi.inference.gradual.typing import Callable
from sympy.physics.units import frequency


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

def plot_force_coefficient(data_array: np.ndarray, ylabel: str, ylim: tuple[float, float], secax_functions_tuple: tuple[Any,Any], filenamebase, save_timeseries = False):
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
    #TODO: not implemented yet...

    # previous version was:
        #     ax.set_ylim((drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2].min() * 0.5,
        #                  drag_coefficient[int(drag_coefficient.shape[0] * periodic_start - 1):, 2].max() * 1.2))
    # ... not applicable for lift-coefficient

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

def analyze_periodic_timeseries(data_array: np.ndarray, periodic_start_rel: float,prominence: float, name: str = "periodic timeseries", outdir=None, pu_per_step = None, verbose=False):
    values_periodic = data_array[int(data_array.shape[0] * periodic_start_rel - 1):, 2]
    steps_LU_periodic = data_array[int(data_array.shape[0] * periodic_start_rel - 1):, 0]
    mean_periodcorrected = None
    max_mean = None
    min_mean = None

    try:
        peaks_max = find_peaks(values_periodic, prominence=prominence) # drag-prominence: ((values.max() - values.min()) / 2); lift-prominence: (lift1100_1500[:,2].max()+lift1100_1500[:,2].min())/2); oder lift: lift_converged[:,2].max()*0.5
        peaks_min = find_peaks(-values_periodic, prominence=prominence)
        if peaks_min[0].shape[0] - peaks_max[0].shape[0] > 0:
            peak_number = peaks_max[0].shape[0]
        else:
            peak_number = peaks_min[0].shape[0]
        if peaks_min[0] < peaks_max[0]:
            first_peak = peaks_min[0]
            last_peak = peaks_max[peak_number - 1]
        else:
            first_peak = peaks_max[0]
            last_peak = peaks_min[peak_number - 1]

        if verbose:
            peak_max_y = values_periodic[peaks_max[0]]
            peak_max_x = steps_LU_periodic[peaks_max[0]]
            peak_min_y = values_periodic[peaks_min[0]]
            peak_min_x = steps_LU_periodic[peaks_min[0]]

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
            plt.plot(steps_LU_periodic, steps_LU_periodic, steps_LU_periodic,
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