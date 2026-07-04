"""
Shared runner for measuring the dynamical critical exponent from TDA
observables of hydrodynamic normal-mode height profiles across four rates
matrices / universality-class regimes.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import git_root
import numpy as np
import matplotlib.pyplot as plt

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
Convert one normal-mode height time series into one scalar TDA time series.
"""
def tda_time_series_from_mode(H, mode_index, observable_name, patch_window, patch_stride=1, tda_every=1):
    
    observable_function = globals()[observable_name]

    n_times = H.shape[0]
    tda_indices = np.arange(0, n_times, tda_every)
    series = np.zeros(len(tda_indices), dtype=float)

    for j, t in enumerate(tda_indices):
        h_profile = H[t, :, mode_index]
        points = patch_point_cloud(h_profile, window=patch_window, stride=patch_stride)
        series[j] = observable_function(points)

    return series

"""
One simulation run at one L and one rates matrix.
"""
def single_run(args):
    
    L, rates_matrix, run_id, observable_name, n_samples, sample_every, patch_divisor, patch_stride, tda_every = args

    process = MultiSpeciesExclusionProcess(dimension=3, density=[1 / 3, 1 / 3, 1 / 3], length=int(L), shuffle=True, rates_matrix=rates_matrix)
    H = process.normal_mode_height_time_series(n_samples=n_samples, sample_every=sample_every)

    patch_window = max(8, int(L) // patch_divisor)

    series_0 = tda_time_series_from_mode(H, mode_index=0, observable_name=observable_name, patch_window=patch_window, patch_stride=patch_stride, tda_every=tda_every)
    series_1 = tda_time_series_from_mode(H, mode_index=1, observable_name=observable_name, patch_window=patch_window, patch_stride=patch_stride, tda_every=tda_every)

    C0 = autocorrelation(series_0)
    C1 = autocorrelation(series_1)

    times = np.arange(len(C0)) * sample_every * tda_every

    tau_0 = relaxation_time(C0, times)
    tau_1 = relaxation_time(C1, times)

    return tau_0, tau_1

"""
Fit tau ~ L^z, so log(tau) = z log(L) + b.
"""
def fit_loglog(L_values, taus):
    
    L_values = np.asarray(L_values, dtype=float)
    taus = np.asarray(taus, dtype=float)

    valid = np.isfinite(taus) & (taus > 0)
    log_L = np.log(L_values[valid])
    log_tau = np.log(taus[valid])

    if len(log_L) < 2:
        return log_L, log_tau, np.nan, np.full_like(log_L, np.nan)

    z, intercept = np.polyfit(log_L, log_tau, 1)
    fit = z * log_L + intercept

    return log_L, log_tau, z, fit

"""
Runs the full checkpointed four-regime DCE calculation and save the 2x2 plot.
"""
def run(observable_name, process_name, output_filename, suptitle=None, load_previous_slurm_job=True):
    
    L_values = np.arange(240, 600, 15)
    N_runs = 24
    n_samples = 6000
    sample_every = 25
    patch_divisor = 8
    patch_stride = 1
    tda_every = 2

    checkpoint_dir = f"{git_root.git_root()}/data/slurm_jobs"
    checkpoint = LCheckpoint(process_name=process_name, L_values=L_values, n_outputs=2 * len(RATES_MATRICES), output_dir=checkpoint_dir)

    if not load_previous_slurm_job:
        checkpoint.delete()
        checkpoint = LCheckpoint(process_name=process_name, L_values=L_values, n_outputs=2 * len(RATES_MATRICES), output_dir=checkpoint_dir)

    checkpoint.install_signal_handlers()

    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    max_workers = min(N_runs, slurm_cpus)

    print(f"observable = {observable_name}")
    print(f"process_name = {process_name}")
    print(f"checkpoint = {checkpoint.filename}")
    print(f"Using max_workers = {max_workers}")
    checkpoint.print_status()

    for L in checkpoint.remaining_L_values():
        print("running L =", L)

        row_values = []

        for rates_index, rates_matrix in enumerate(RATES_MATRICES):
            print(f"  rates matrix {rates_index + 1}/{len(RATES_MATRICES)}")

            args_list = [(L, rates_matrix, run_id, observable_name, n_samples, sample_every, patch_divisor, patch_stride, tda_every) for run_id in range(N_runs)]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(single_run, args_list))

            results = np.asarray(results, dtype=float)

            tau0_mean = np.nanmean(results[:, 0])
            tau1_mean = np.nanmean(results[:, 1])

            row_values.extend([tau0_mean, tau1_mean])

            print(f"    mode 0 tau = {tau0_mean}")
            print(f"    mode 1 tau = {tau1_mean}")

        checkpoint.record(L, row_values, autosave=True)
        checkpoint.print_status()

    tau_table = checkpoint.get_results()

    fig, axis = plt.subplots(2, 2, figsize=(8, 8))

    for rates_index, ax in enumerate(axis.ravel()):
        col0 = 2 * rates_index
        col1 = col0 + 1

        tau0s = tau_table[:, col0]
        tau1s = tau_table[:, col1]

        log_L_0, log_tau_0, z0, fit0 = fit_loglog(L_values, tau0s)
        log_L_1, log_tau_1, z1, fit1 = fit_loglog(L_values, tau1s)

        ax.plot(log_L_0, log_tau_0, "go", alpha=0.5, ms=4, label=fr"mode 0, $z_0 = {z0:.2f}$")
        ax.plot(log_L_1, log_tau_1, "ro", alpha=0.5, ms=4, label=fr"mode 1, $z_1 = {z1:.2f}$")

        ax.plot(log_L_0, fit0, "b--", label="Fit mode 0")
        ax.plot(log_L_1, fit1, "k--", label="Fit mode 1")

        ax.set_title(RATES_TITLES[rates_index])
        ax.set_xlabel("log(L)")
        ax.set_ylabel(r"log($\tau_{\mathrm{TDA}}$)")
        ax.legend(fontsize=8)

    if suptitle is None:
        suptitle = fr"{observable_name} Relaxation Time Scaling"

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()

    output_dir = f"{git_root.git_root()}/data/hydro-mode-tda-dce"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/{output_filename}"
    plt.savefig(output_path, dpi=300)
    print("saved figure to", output_path)
    plt.show()

    if checkpoint.is_complete():
        print("Checkpoint complete:")
        print("  ", checkpoint.filename)

    return L_values, tau_table
