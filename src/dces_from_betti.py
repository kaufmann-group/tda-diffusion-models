import os
import numpy as np
import gudhi as gd
import git_root
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from utils import *

def normal_mode_beta_1s(total_steps, L, target_skip_steps, rates_matrix, r, max_edge_length):
    model = MultiSpeciesExclusionProcess(dimension=3, density=[1/3, 1/3, 1/3], rates_matrix=rates_matrix, length=L, shuffle=False)
    model.set_chain(equal_spread_chain(L).tolist())

    sample_every_sweeps = max(1, int(round(target_skip_steps / L)))
    actual_skip_steps = sample_every_sweeps * L
    n_samples = total_steps // actual_skip_steps + 1

    height_history = model.normal_mode_height_time_series(n_samples=n_samples, sample_every=sample_every_sweeps)

    n_modes = height_history.shape[2]
    beta_1_values_by_mode = [[] for _ in range(n_modes)]

    x_coords = np.arange(L + 1)

    for sample_index in range(n_samples):
        for mode_index in range(n_modes):
            heights = height_history[sample_index, :, mode_index]
            point_cloud = np.column_stack((x_coords, heights))
            beta_1_value = beta_1(point_cloud, r=r, max_edge_length=max_edge_length)
            beta_1_values_by_mode[mode_index].append(beta_1_value)

    return np.array(beta_1_values_by_mode), actual_skip_steps

if __name__ == "__main__":
    total_steps = 1000000
    target_skip_steps = 1000
    N_runs = 50
    max_workers = min(N_runs, 8)

    r = 1.5
    max_edge_length = 3.0

    rates_matrix = np.array([[0.0, 2.0, 2.0], [1.0, 0.0, 1.5], [1.0, 1.5, 0.0]], dtype=np.float64)

    L_values = np.arange(120, 600, 15)

    output_dir = f"{git_root.git_root()}/data/normal_mode_beta1_relaxations"
    os.makedirs(output_dir, exist_ok=True)

    saturation_times_by_mode = None

    for L in L_values:
        print(f"running L = {L}")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(normal_mode_beta_1s, total_steps, L, target_skip_steps, rates_matrix, r, max_edge_length) for _ in range(N_runs)]
            results = [f.result() for f in futures]

        actual_skip_steps = results[0][1]
        mode_results = np.array([result[0] for result in results])

        n_modes = mode_results.shape[1]
        n_samples = mode_results.shape[2]
        sampled_times = np.arange(n_samples) * actual_skip_steps

        if saturation_times_by_mode is None:
            saturation_times_by_mode = [[] for _ in range(n_modes)]

        for mode_index in range(n_modes):
            mode_ensemble = mode_results[:, mode_index, :]
            mode_mean = np.mean(mode_ensemble, axis=0)
            mode_smooth = smooth(mode_mean)

            tau, sat_value = saturation_time(sampled_times, mode_smooth)
            saturation_times_by_mode[mode_index].append(tau)

            print(f"L = {L}, mode = {mode_index + 1}, saturation time = {tau}, saturation value = {sat_value}")

            plt.figure(figsize=(8, 4))
            plt.plot(sampled_times, mode_mean, alpha=0.35, label="raw ensemble mean")
            plt.plot(sampled_times, mode_smooth, linewidth=2, label="smoothed ensemble mean")

            if not np.isnan(tau):
                plt.axvline(tau, linestyle="--", label=fr"$\tau = {tau:.0f}$")

            plt.xlabel("monte carlo steps")
            plt.ylabel(fr"$\langle \beta_1^{{({mode_index + 1})}}(r,t) \rangle$")
            plt.title(f"normal-mode beta 1 relaxation, mode={mode_index + 1}, L={L}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/mode_{mode_index + 1}_beta1_relaxation_L_{L}.png", dpi=300)
            plt.close()

    saturation_times_by_mode = [np.array(times, dtype=float) for times in saturation_times_by_mode]

    for mode_index, saturation_times in enumerate(saturation_times_by_mode):
        z, intercept, log_L, log_tau = fit_dynamic_exponent(L_values, saturation_times)
        fit_line = z * log_L + intercept

        plt.figure(figsize=(7, 5))
        plt.scatter(log_L, log_tau, alpha=0.65)
        plt.plot(log_L, fit_line, linestyle="--", linewidth=2, label=fr"$z_{{{mode_index + 1}}} = {z:.3f}$")

        plt.xlabel(r"$\log L$", fontsize=14)
        plt.ylabel(r"$\log \tau(L)$", fontsize=14)
        plt.title(fr"Dynamic exponent from normal-mode $\beta_1$, mode {mode_index + 1}", fontsize=16)

        plt.grid(True, linestyle=":", linewidth=1)
        plt.legend(fontsize=13)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/dynamic_exponent_normal_mode_{mode_index + 1}_beta1_log_fit.png", dpi=300)
        plt.close()

        print(f"mode {mode_index + 1}: z = {z}")