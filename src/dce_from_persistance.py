import numpy as np
import gudhi as gd
import git_root
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from utils import *

def p_maxs(steps, L, skip, rates_matrix, max_edge_length):
    model = MultiSpeciesExclusionProcess(dimension=3, density=[1/3, 1/3, 1/3], rates_matrix=rates_matrix, length=L, shuffle=False)
    model.set_chain(equal_spread_chain(L).tolist())

    path_history = model.simulate(steps=steps, store_history=True, get_projection=True, skip=skip)
    p_max_values = []

    for path in path_history:
        p_max_value = p_max(path, max_edge_length=max_edge_length)
        p_max_values.append(p_max_value)

    return np.array(p_max_values)

if __name__ == "__main__":
    steps = 1000000
    skip = 100
    N_runs = 50

    max_edge_length = 3.0

    rates_matrix = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float64)

    sampled_times = np.arange(0, steps + 1, skip)
    L_values = np.arange(60, 600, 60)
    saturation_times = []

    for L in L_values:
        with ProcessPoolExecutor(max_workers=N_runs) as executor:
            futures = [executor.submit(p_maxs, steps, L, skip, rates_matrix, max_edge_length) for _ in range(N_runs)]
            results = [f.result() for f in futures]

        p_max_ensemble = np.array(results)
        p_max_mean = np.mean(p_max_ensemble, axis=0)
        p_max_smooth = smooth(p_max_mean)

        tau, sat_value = saturation_time(sampled_times, p_max_smooth)
        saturation_times.append(tau)

        print(f"L = {L}, saturation time = {tau}, saturation value = {sat_value}")

        plt.figure(figsize=(8, 4))
        plt.plot(sampled_times, p_max_mean, alpha=0.35, label="raw ensemble mean")
        plt.plot(sampled_times, p_max_smooth, linewidth=2, label="smoothed ensemble mean")
        plt.axvline(tau, linestyle="--", label=fr"$\tau = {tau:.0f}$")
        plt.xlabel("monte carlo steps")
        plt.ylabel(r"$\langle P_{\max}(t) \rangle$")
        plt.title(f"p max relaxation, L={L}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{git_root.git_root()}/data/pmax_relaxations_attempt_1/pmax_relaxation_L_{L}.png", dpi=300)
        plt.close()

    saturation_times = np.array(saturation_times)

    z, intercept, log_L, log_tau = fit_dynamic_exponent(L_values, saturation_times)
    fit_line = z * log_L + intercept

    plt.figure(figsize=(7, 5))
    plt.scatter(log_L, log_tau, alpha=0.65)
    plt.plot(log_L, fit_line, linestyle="--", linewidth=2, label=fr"$z_{{P_{{\max}}}} = {z:.3f}$")

    plt.xlabel(r"$\log L$", fontsize=14)
    plt.ylabel(r"$\log \tau(L)$", fontsize=14)
    plt.title(r"Dynamic exponent from $P_{\max}$", fontsize=18)

    plt.grid(True, linestyle=":", linewidth=1)
    plt.legend(fontsize=13)

    plt.tight_layout()
    plt.savefig(f"{git_root.git_root()}/data/pmax_relaxations_attempt_1/dynamic_exponent_pmax_log_fit.png", dpi=300)