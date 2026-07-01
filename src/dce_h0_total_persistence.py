"""
plots the dynamical critical exponent for different rates matrices
representing different combinations of the KPZ and diffusive universality
classes, using H0 persistence of hydrodynamic normal mode time series.
"""

import git_root
import numpy as np
import matplotlib.pyplot as plt

import os
from concurrent.futures import ProcessPoolExecutor

from ..utils import *

"""
convert a hydrodynamic normal mode height time series into an H0 persistence time series.
"""
def h0_time_series_from_mode(H, mode_index, patch_window, patch_stride=1, tda_every=1):
    n_times = H.shape[0]

    tda_indices = np.arange(0, n_times, tda_every)
    h0_series = np.zeros(len(tda_indices), dtype=float)

    for j, t in enumerate(tda_indices):
        h_profile = H[t, :, mode_index]

        points = patch_point_cloud(h_profile, window=patch_window, stride=patch_stride)

        h0_series[j] = h0_total_persistence(points)

    return h0_series

def single_run(args):
    L, rates_matrix, run_id = args

    n_samples = 6000
    sample_every = 25
    patch_divisor = 8
    patch_stride = 1

    tda_every = 2

    process = MultiSpeciesExclusionProcess(dimension=3, density=[1/3, 1/3, 1/3], length=int(L), shuffle=True, rates_matrix=rates_matrix)

    H = process.normal_mode_height_time_series(n_samples=n_samples, sample_every=sample_every)

    patch_window = max(8, int(L) // patch_divisor)

    h0_series_1 = h0_time_series_from_mode(H, mode_index=0, patch_window=patch_window, patch_stride=patch_stride, tda_every=tda_every)
    h0_series_2 = h0_time_series_from_mode(H, mode_index=1, patch_window=patch_window, patch_stride=patch_stride, tda_every=tda_every)

    C1 = autocorrelation(h0_series_1)
    C2 = autocorrelation(h0_series_2)

    times = np.arange(len(C1)) * sample_every * tda_every

    t1 = relaxation_time(C1, times)
    t2 = relaxation_time(C2, times)

    return t1, t2

def getzs(rates_matrix):
    tau1s = []
    tau2s = []

    L_values = np.arange(240, 600, 15)

    N_runs = 24

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    max_workers = min(N_runs, slurm_cpus)

    print(f"Using max_workers = {max_workers}")

    for L in L_values:
        print("running L =", L)

        args_list = [(L, rates_matrix, run_id) for run_id in range(N_runs)]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(single_run, args_list))

        results = np.asarray(results, dtype=float)

        tau1_runs = results[:, 0]
        tau2_runs = results[:, 1]

        tau1_mean = np.nanmean(tau1_runs)
        tau2_mean = np.nanmean(tau2_runs)

        tau1s.append(tau1_mean)
        tau2s.append(tau2_mean)

    tau1s = np.asarray(tau1s, dtype=float)
    tau2s = np.asarray(tau2s, dtype=float)

    valid1 = np.isfinite(tau1s) & (tau1s > 0)
    valid2 = np.isfinite(tau2s) & (tau2s > 0)

    log_L_1 = np.log(L_values[valid1])
    log_L_2 = np.log(L_values[valid2])

    log_t1 = np.log(tau1s[valid1])
    log_t2 = np.log(tau2s[valid2])

    z1, intercept1 = np.polyfit(log_L_1, log_t1, 1)
    z2, intercept2 = np.polyfit(log_L_2, log_t2, 1)

    fit1 = z1 * log_L_1 + intercept1
    fit2 = z2 * log_L_2 + intercept2

    return log_L_1, log_L_2, log_t1, log_t2, z1, z2, fit1, fit2

def plot(ax, rates_matrix):
    log_L_1, log_L_2, log_t1, log_t2, z1, z2, fit1, fit2 = getzs(rates_matrix)

    ax.plot(log_L_1, log_t1, "go", alpha=0.5, ms=4, label=fr"mode 0, $z_1 = {z1:.2f}$")

    ax.plot(log_L_2, log_t2, "ro", alpha=0.5, ms=4, label=fr"mode 1, $z_2 = {z2:.2f}$")

    ax.plot(log_L_1, fit1, "b--", label="Fit mode 0")
    ax.plot(log_L_2, fit2, "k--", label="Fit mode 1")

    ax.set_xlabel("log(L)")
    ax.set_ylabel(r"log($\tau_{\mathrm{TDA}}$)")
    ax.legend(fontsize=8)

if __name__ == "__main__":
    fig, axis = plt.subplots(2, 2, figsize=(8, 8))

    rates_matrix_1 = np.array([
        [0.0, 0.1, 1.1],
        [2.1, 0.0, 3.1],
        [0.1, 0.1, 0.0],
    ])

    rates_matrix_2 = np.array([
        [0.0, 1.0, 0.1],
        [1.0, 0.0, 0.1],
        [2.1, 2.1, 0.0],
    ])

    rates_matrix_3 = np.array([
        [0.0, 0.1, 1.0],
        [2.1, 0.0, 2.1],
        [1.0, 0.1, 0.0],
    ])

    rates_matrix_4 = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])

    """
    plotting
    """

    plot(ax=axis[0, 0], rates_matrix=rates_matrix_1)
    axis[0, 0].set_title("Rates matrix 1")
    plot(ax=axis[0, 1], rates_matrix=rates_matrix_2)
    axis[0, 1].set_title("Rates matrix 2")
    plot(ax=axis[1, 0], rates_matrix=rates_matrix_3)
    axis[1, 0].set_title("Rates matrix 3")
    plot(ax=axis[1, 1], rates_matrix=rates_matrix_4)
    axis[1, 1].set_title("Rates matrix 4")

    fig.suptitle(r"$H_0$ Persistence Relaxation Time Scaling for Different Rates Matrices", fontsize=14,)

    plt.tight_layout()

    plt.savefig(f"{git_root.git_root()}/data/tda_h0_timescaling.png", dpi=300)
    plt.show()