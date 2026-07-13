"""
the goal of this program is to demonstrate the multi species 
exclusion process falls within the the KPZ universality class.
"""

import git_root
import numpy as np
import matplotlib.pyplot as plt

from utils import *

def get_dynamical_critical_exponent(species_size):
    taus = [[] for _ in range(species_size-1)]
    L_values = np.arange(10+(species_size-10%species_size), 300+(species_size-300%species_size), species_size)

    for L in L_values:
        dimension = species_size
        density = np.full(dimension, 1.0 / dimension, dtype=np.float64)

        rates_matrix = np.triu(np.ones((dimension, dimension), dtype=np.float64), k=1)

        process = MultiSpeciesExclusionProcess(dimension=dimension, density=density, rates_matrix=rates_matrix, length=L)

        X = process.fourier_time_series(n_samples=30000, sample_every=1)

        for i in range(species_size-1):
            taus[i].append(relaxation_time(autocorrelation(X[:, i])))

    results = []
    for i in range(species_size-1):
        taus_i = np.array(taus[i])
        valid = np.isfinite(taus_i) & (taus_i > 0)

        logL = np.log(L_values[valid])
        logtau = np.log(taus_i[valid])   

        z, intercept = np.polyfit(logL, logtau, 1)
        fit = intercept + z * logL

        results.append((logL, logtau, fit, z))

    return results

if __name__ == "__main__":
    """
    solving for the critical dynamical exponent for n = 2, 3, 4, 5, 6, 7, 8, and 9 species
    """

    fig, axes = plt.subplots(4, 2, figsize=(10, 15)) 
    fig.suptitle("the critical dynamical exponent for many species")

    for species_size, ax in zip(np.arange(2, 10, 1), axes.flatten()): 
        results = get_dynamical_critical_exponent(species_size=species_size) 

        for i, result in enumerate(results):  
            logL, logtau, fit, z = result
            
            line, = ax.plot(logL, logtau, "o", alpha=0.6)
            current_color = line.get_color()
            ax.plot(logL, fit, "--", color=current_color, label=fr"$z_{{{i+1}}} = {z:.3f}$")
            
        ax.set_xlabel(r"$\log L$")
        ax.set_ylabel(r"$\log \tau(L)$")
        ax.set_title(f"{species_size} species")
        ax.legend(loc="upper left", fontsize='small')
        ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout() 

    plt.savefig(f"{git_root.git_root()}/data/fourier-dce/many_species_dynamical_critical_exponent_simulation.png", dpi=300)
    