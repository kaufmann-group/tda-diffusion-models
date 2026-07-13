"""
Shared runner for measuring the dynamical critical exponent from TDA
observables of the raw species projection point cloud across two rates
matrices: asymmetric and symmetric two-species exclusion processes.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import git_root
import numpy as np
import matplotlib.pyplot as plt

from src.utils import *


RATES_MATRICES = [
    np.array([
        [0.0, 1.0],
        [0.0, 0.0],
    ]),
    np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ]),
]

RATES_TITLES = ["Asymmetric Two-Species Process","Symmetric Two-Species Process"]


"""
Convert one raw projection time series into one scalar TDA time series.
"""
def tda_time_series(P, observable_name, tda_every=1):
    
    observable_function = globals()[observable_name]

    n_times = P.shape[0]
    tda_indices = np.arange(0, n_times, tda_every)
    series = np.zeros(len(tda_indices), dtype=float)

    for j, t in enumerate(tda_indices):
        points = P[t, :, :]
        series[j] = observable_function(points)

    return series


"""
Generate a time series of raw projected MSEP paths.
"""
def raw_projection_time_series(process, L, n_samples, sample_every):
    steps_between_samples = int(sample_every) * int(L)
    steps_total = (int(n_samples) - 1) * steps_between_samples

    P = process.simulate(steps=steps_total, store_history=True, get_projection=True, skip=steps_between_samples)

    return np.asarray(P, dtype=float)


"""
One simulation run at one L and one rates matrix.
"""
def single_run(args):
    
    L, rates_matrix, run_id, observable_name, n_samples, sample_every, tda_every = args

    # seed = 12345 + 100000 * int(L) + int(run_id)
    process = MultiSpeciesExclusionProcess(dimension=2, density=[1/2, 1/2], length=int(L), shuffle=True, rates_matrix=rates_matrix)

    P = raw_projection_time_series(process=process, L=L, n_samples=n_samples, sample_every=sample_every)

    series = tda_time_series(P=P, observable_name=observable_name, tda_every=tda_every)

    C = autocorrelation(series)
    times = np.arange(len(C)) * sample_every * tda_every
    tau = relaxation_time(C, times)

    return tau


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
Runs the full checkpointed two-regime DCE calculation and save the plot.
"""
def run(observable_name, process_name, output_filename, suptitle=None, load_previous_slurm_job=True):
    
    L_values = np.arange(240, 720, 20)
    N_runs = 24
    n_samples = 6000
    sample_every = 25
    tda_every = 2

    checkpoint_dir = f"{git_root.git_root()}/data/slurm_jobs"
    checkpoint = LCheckpoint(process_name=process_name, L_values=L_values, n_outputs=len(RATES_MATRICES), output_dir=checkpoint_dir)

    if not load_previous_slurm_job:
        checkpoint.delete()
        checkpoint = LCheckpoint(process_name=process_name, L_values=L_values, n_outputs=len(RATES_MATRICES), output_dir=checkpoint_dir)

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

            args_list = [(L, rates_matrix, run_id, observable_name, n_samples, sample_every, tda_every) for run_id in range(N_runs)]

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(single_run, args_list))

            results = np.asarray(results, dtype=float)

            tau_mean = np.nanmean(results[:])

            row_values.extend([tau_mean])

            print(f"    tau = {tau_mean}")

        checkpoint.record(L, row_values, autosave=True)
        checkpoint.print_status()

    tau_table = checkpoint.get_results()

    fig, axis = plt.subplots(1, 2, figsize=(10, 4))

    for rates_index, ax in enumerate(axis.ravel()):
        col = rates_index

        taus = tau_table[:, col]

        log_L, log_tau, z, fit = fit_loglog(L_values, taus)

        ax.plot(log_L, log_tau, "go", alpha=0.5, ms=4, label=fr"$z = {z:.2f}$")

        ax.plot(log_L, fit, "b--", label="Fit")

        ax.set_title(RATES_TITLES[rates_index])
        ax.set_xlabel("log(L)")
        ax.set_ylabel(r"log($\tau_{\mathrm{TDA}}$)")
        ax.legend(fontsize=8)

    if suptitle is None:
        suptitle = fr"{observable_name} Relaxation Time Scaling"

    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout()

    output_dir = f"{git_root.git_root()}/data/proj-tda-dce"
    os.makedirs(output_dir, exist_ok=True)

    output_path = f"{output_dir}/{output_filename}"
    plt.savefig(output_path, dpi=300)
    print("saved figure to", output_path)
    plt.show()

    if checkpoint.is_complete():
        print("Checkpoint complete:")
        print("  ", checkpoint.filename)

    return L_values, tau_table