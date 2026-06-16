import numpy as np
import gudhi as gd
import git_root
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from utils import *

def beta_1s(steps, L, skip, rates_matrix, r, max_edge_length):
    model = MultiSpeciesExclusionProcess(dimension=3, density=[1/3, 1/3, 1/3], rates_matrix=rates_matrix, length=L, shuffle=False)
    model.set_chain(equal_spread_chain(L).tolist())

    path_history = model.simulate(steps=steps, store_history=True, get_projection=True, skip=skip)
    beta_1_values = []

    for path in path_history:
        beta_1_value = beta_1(path, r=r, max_edge_length=max_edge_length)
        beta_1_values.append(beta_1_value)

    return np.array(beta_1_values)

if __name__ == "__main__":
    steps = 1000000
    skip = 1000
    N_runs = 50

    r = 1.5
    max_edge_length = 3.0

    rates_matrix = np.array([[0.0, 2.0, 2.0], [1.0, 0.0, 1.5], [1.0, 1.5, 0.0]], dtype=np.float64)

    sampled_times = np.arange(0, steps + 1, skip)
    L_values = np.arange(120, 600, 15)
    saturation_times = []

    for L in L_values: 
        with ProcessPoolExecutor(max_workers=N_runs) as executor: 
            futures = [executor.submit(beta_1s, steps, L, skip, rates_matrix, r, max_edge_length) for _ in range(N_runs)] 
            results = [f.result() for f in futures] 
            
        beta_1_ensemble = np.array(results)
        beta_1_mean = np.mean(beta_1_ensemble, axis=0)
        beta_1_smooth = smooth(beta_1_mean)

        tau, sat_value = saturation_time(sampled_times, beta_1_smooth)
        saturation_times.append(tau)

        print(f"L = {L}, saturation time = {tau}, saturation value = {sat_value}")

        plt.figure(figsize=(8, 4))
        plt.plot(sampled_times, beta_1_mean, alpha=0.35, label="raw ensemble mean")
        plt.plot(sampled_times, beta_1_smooth, linewidth=2, label="smoothed ensemble mean")
        plt.axvline(tau, linestyle="--", label=fr"$\tau = {tau:.0f}$")
        plt.xlabel("monte carlo steps")
        plt.ylabel(r"$\langle \beta_1(r,t) \rangle$")
        plt.title(f"beta 1 relaxation, L={L}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{git_root.git_root()}/data/beta1_relaxations/beta1_relaxation_L_{L}.png", dpi=300)
        plt.close()

    saturation_times = np.array(saturation_times)

    z, intercept, log_L, log_tau = fit_dynamic_exponent(L_values, saturation_times)
    fit_line = z * log_L + intercept

    plt.figure(figsize=(7, 5))
    plt.scatter(log_L, log_tau, alpha=0.65)
    plt.plot(log_L, fit_line, linestyle="--", linewidth=2, label=fr"$z_1 = {z:.3f}$")

    plt.xlabel(r"$\log L$", fontsize=14)
    plt.ylabel(r"$\log \tau(L)$", fontsize=14)
    plt.title(r"Dynamic exponent from $\beta_1$", fontsize=18)

    plt.grid(True, linestyle=":", linewidth=1)
    plt.legend(fontsize=13)

    plt.tight_layout()
    plt.savefig(f"{git_root.git_root()}/data/beta1_relaxations/dynamic_exponent_beta1_log_fit.png", dpi=300)