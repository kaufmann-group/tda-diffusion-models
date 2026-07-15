"""
measuring the dynamical critical exponent from TDA observables across four universality class regimes
"""

import os

import git_root
import ipyparallel as ipp
import matplotlib.pyplot as plt
import numpy as np

from src.utils import *

RATES_MATRICES = [
    np.array([
        [0.0, 0.1, 1.1],
        [2.1, 0.0, 3.1],
        [0.1, 0.1, 0.0],
    ]),
    np.array([
        [0.0, 1.0, 0.1],
        [1.0, 0.0, 0.1],
        [2.1, 2.1, 0.0],
    ]),
    np.array([
        [0.0, 0.1, 1.0],
        [2.1, 0.0, 2.1],
        [1.0, 0.1, 0.0],
    ]),
    np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]),
]

RATES_TITLES = ["Both Modes KPZ", "One Mode Diffusive, One Mode KPZ", "One Mode KPZ, One Mode Diffusive", "Both Modes Diffusive"]

"""
convert one normal mode height time series into one scalar TDA time series
"""
def tda_time_series_from_mode(H, mode_index, observable_name, patch_window, patch_stride=1, tda_every=1):
    import numpy as np
    import src.utils as utils

    observable_function = getattr(utils, observable_name)

    n_times = H.shape[0]
    tda_indices = np.arange(0, n_times, tda_every)
    series = np.zeros(len(tda_indices), dtype=float)

    for j, time_index in enumerate(tda_indices):
        height_profile = H[time_index, :, mode_index]
        points = utils.patch_point_cloud(height_profile, window=patch_window, stride=patch_stride)
        series[j] = observable_function(points)

    return series

"""
compute one simulation run for one system size, rates matrix, and run index
"""
def single_run(args):
    import numpy as np
    import src.utils as utils

    L, rates_index, rates_matrix, run_id, observable_name, n_samples, sample_every, patch_divisor, patch_stride, tda_every = args

    process = utils.MultiSpeciesExclusionProcess(dimension=3, density=[1 / 3, 1 / 3, 1 / 3], length=int(L), shuffle=True, rates_matrix=rates_matrix)
    H = process.normal_mode_height_time_series(n_samples=n_samples, sample_every=sample_every)

    patch_window = max(8, int(L) // patch_divisor)

    series_0 = tda_time_series_from_mode(H=H, mode_index=0, observable_name=observable_name, patch_window=patch_window, patch_stride=patch_stride, tda_every=tda_every)
    series_1 = tda_time_series_from_mode(H=H, mode_index=1, observable_name=observable_name, patch_window=patch_window, patch_stride=patch_stride, tda_every=tda_every)

    C0 = utils.autocorrelation(series_0)
    C1 = utils.autocorrelation(series_1)

    times = np.arange(len(C0), dtype=float) * sample_every * tda_every

    tau_0 = utils.relaxation_time(C0, times)
    tau_1 = utils.relaxation_time(C1, times)

    return int(L), int(rates_index), int(run_id), float(tau_0), float(tau_1)

"""
average independently computed relaxation times over all runs
"""
def aggregate_results(results, n_runs):
    row_values = []

    for rates_index in range(len(RATES_MATRICES)):
        rates_results = [result for result in results if result[1] == rates_index]

        if len(rates_results) != n_runs:
            raise RuntimeError(f"Rates matrix {rates_index} returned {len(rates_results)} runs; expected {n_runs}.")

        rates_results.sort(key=lambda result: result[2])

        tau_0_values = np.asarray([result[3] for result in rates_results], dtype=float)
        tau_1_values = np.asarray([result[4] for result in rates_results], dtype=float)

        tau_0_mean = np.nanmean(tau_0_values)
        tau_1_mean = np.nanmean(tau_1_values)

        row_values.extend([tau_0_mean, tau_1_mean])

        print(f"    rates matrix {rates_index + 1}: mode 0 tau = {tau_0_mean}")
        print(f"    rates matrix {rates_index + 1}: mode 1 tau = {tau_1_mean}")

    return row_values

"""
Produce and save the four-regime dynamical critical exponent plot.
"""
def plot_results(L_values, tau_table, observable_name, output_filename, suptitle):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    for rates_index, ax in enumerate(axes.ravel()):
        mode_0_column = 2 * rates_index
        mode_1_column = mode_0_column + 1

        tau_0_values = tau_table[:, mode_0_column]
        tau_1_values = tau_table[:, mode_1_column]

        log_L_0, log_tau_0, z_0, fit_0 = fit_loglog(L_values, tau_0_values)
        log_L_1, log_tau_1, z_1, fit_1 = fit_loglog(L_values, tau_1_values)

        ax.plot(log_L_0, log_tau_0, "go", alpha=0.5, ms=4, label=fr"mode 0, $z_0={z_0:.2f}$")
        ax.plot(log_L_1, log_tau_1, "ro", alpha=0.5, ms=4, label=fr"mode 1, $z_1={z_1:.2f}$")

        ax.plot(log_L_0, fit_0, "b--", label="Fit mode 0")
        ax.plot(log_L_1, fit_1, "k--", label="Fit mode 1")

        ax.set_title(RATES_TITLES[rates_index])
        ax.set_xlabel(r"$\log L$")
        ax.set_ylabel(r"$\log \tau_{\mathrm{TDA}}$")
        ax.legend(fontsize=8)

    if suptitle is None:
        suptitle = fr"{observable_name} Relaxation Time Scaling"

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()

    output_dir = f"{git_root.git_root()}/data/hydro-mode-tda-dce"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/{output_filename}"

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print("saved figure to", output_path)
    plt.show()