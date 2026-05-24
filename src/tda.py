from msep.msep_cpp import MultiSpeciesExclusionProcess

import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt

from utils.tda import tda_observables

"""
runs none exclusion process trajectory then compute tda observables at certain times
"""
def single_tda_trajectory(steps, L, skip, rates_matrix, r, epsilon, max_edge_length):
    
    model = MultiSpeciesExclusionProcess(dimension=3, density=[1/3, 1/3, 1/3], rates_matrix=rates_matrix, length=L, shuffle=False)
    #model = MultiSpeciesExclusionProcess(dimension=3, density=[1/3, 1/3, 1/3], rates_matrix=rates_matrix, length=L, shuffle=False)


    path_history = model.simulate(steps=steps, store_history=True, get_projection=True)

    saved_paths = path_history[::skip]

    beta_1_values, p_max_values, p_total_values, n_epsilon_values = [], [], [], []

    for path in saved_paths:
        beta_1, p_max, p_total, n_epsilon = tda_observables(path, r=r, epsilon=epsilon, max_edge_length=max_edge_length)

        beta_1_values.append(beta_1)
        p_max_values.append(p_max)
        p_total_values.append(p_total)
        n_epsilon_values.append(n_epsilon)

    return np.array(beta_1_values), np.array(p_max_values), np.array(p_total_values), np.array(n_epsilon_values)

if __name__ == "__main__":
    """
    <beta_1> = the ensemble averaged number of loops alive at the fixed filtration scale r 
        at scale r = 1.5 how many loops does the projected path typically have at time?

    <P_max> = the ensemble average of the largest h1 persistence in each snapshot
        what is the strength of the most prominent loop in the projected path at time t?

    <P_tot> = the ensemble average of the sum of all h1 bar lengths
        how much total loop structure exists in the projected path at time?

    <N_e> = the ensemble averaged number of h1 loops whose persistence is larger than epsilon
        how many loops are significant enough to survive beyond the noise threshold epsilon? 
    """

    steps = 35000
    L = 300
    skip = 10
    N_runs = 50

    r = 1.5
    epsilon = 0.2
    max_edge_length = 5.0

    rates_matrix = np.array(
        [
            [0.0, 2.0, 2.0],
            [1.0, 0.0, 1.5],
            [1.0, 1.5, 0.0],
        ],
        dtype=np.float64,
    )

    sampled_times = np.arange(0, steps + 1, skip)
    beta_1_ensemble, p_max_ensemble, p_total_ensemble, n_epsilon_ensemble = [], [], [], []

    for run in range(N_runs):
        print(f"running ensemble number: {run + 1}")

        beta_1_values, p_max_values, p_total_values, n_epsilon_values = single_tda_trajectory(steps=steps, L=L, skip=skip, rates_matrix=rates_matrix, r=r, epsilon=epsilon, max_edge_length=max_edge_length)
        
        beta_1_ensemble.append(beta_1_values)
        p_max_ensemble.append(p_max_values)
        p_total_ensemble.append(p_total_values)
        n_epsilon_ensemble.append(n_epsilon_values)

    beta_1_ensemble = np.array(beta_1_ensemble)    
    p_max_ensemble = np.array(p_max_ensemble)    
    p_total_ensemble = np.array(p_total_ensemble)    
    n_epsilon_ensemble = np.array(n_epsilon_ensemble)    

    # ensamble means
    beta_1_mean = np.mean(beta_1_ensemble, axis=0)
    p_max_mean = np.mean(p_max_ensemble, axis=0)
    p_total_mean = np.mean(p_total_ensemble, axis=0)
    n_epsilon_mean = np.mean(n_epsilon_ensemble, axis=0)

    # standard error calculations
    beta_1_sem = np.std(beta_1_ensemble, axis=0, ddof=1) / np.sqrt(N_runs)
    p_max_sem = np.std(p_max_ensemble, axis=0, ddof=1) / np.sqrt(N_runs)
    p_total_sem = np.std(p_total_ensemble, axis=0, ddof=1) / np.sqrt(N_runs)
    n_epsilon_sem = np.std(n_epsilon_ensemble, axis=0, ddof=1) / np.sqrt(N_runs)

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(sampled_times, beta_1_mean)
    axes[0].fill_between(sampled_times, beta_1_mean - beta_1_sem, beta_1_mean + beta_1_sem, alpha=0.3)
    axes[0].set_ylabel(r"$\langle \beta_1(r,t) \rangle$")
    axes[0].set_title(f"ensamble averaged tda observables, "fr"L={L}, N={N_runs}, r={r}, $\varepsilon$={epsilon}")

    axes[1].plot(sampled_times, p_max_mean)
    axes[1].fill_between(sampled_times, p_max_mean - p_max_sem, p_max_mean + p_max_sem, alpha=0.3)
    axes[1].set_ylabel(r"$\langle P_{\max}(t) \rangle$")

    axes[2].plot(sampled_times, p_total_mean)
    axes[2].fill_between(sampled_times, p_total_mean - p_total_sem, p_total_mean + p_total_sem, alpha=0.3)
    axes[2].set_ylabel(r"$\langle P_{\mathrm{total}}(t) \rangle$")

    axes[3].plot(sampled_times, n_epsilon_mean)
    axes[3].fill_between(sampled_times, n_epsilon_mean - n_epsilon_sem, n_epsilon_mean + n_epsilon_sem, alpha=0.3)
    axes[3].set_ylabel(r"$\langle N_\epsilon(t) \rangle$")
    axes[3].set_xlabel("monte carlo steps")

    plt.tight_layout()
    plt.savefig("figures/tda_ensemble_statistics.png", dpi=300)
    plt.show()