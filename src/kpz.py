"""
the goal of this program is to demonstrate the multi species 
exclusion process falls within the the KPZ universality class.
"""

from msep.msep_cpp import MultiSpeciesExclusionProcess

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from utils.autocorrelation import autocorrelation
from utils.relaxation_time import get_relaxation_time

def get_dynamical_critical_exponent(species_size):
    taus = []
    L_values = np.arange(10+(species_size-10%species_size), 300+(species_size-300%species_size), species_size)

    for L in L_values:
        dimension = species_size
        density = np.full(dimension, 1.0 / dimension, dtype=np.float64)

        rates_matrix = np.triu(np.ones((dimension, dimension), dtype=np.float64), k=1)

        #process = MultiSpeciesExclusionProcess(dimension=dimension, density=density, rates_matrix=rates_matrix, length=L, seed=2504+L)
        process = MultiSpeciesExclusionProcess(dimension=dimension, density=density, rates_matrix=rates_matrix, length=L)

        X = process.fourier_time_series(n_samples=60000, species=0, sample_every=1)
        C = autocorrelation(X)

        taus.append(get_relaxation_time(C, L))

    taus = np.array(taus)
    valid = np.isfinite(taus) & (taus > 0)

    logL = np.log(L_values[valid])
    logtau = np.log(taus[valid])      

    z, intercept = np.polyfit(logL, logtau, 1)
    fit = intercept + z * logL

    return logL, logtau, fit, z

if __name__ == "__main__":
    """
    solving for the critical dynamical exponent for 3 species
    """

    logL, logtau, fit, z = get_dynamical_critical_exponent(species_size=3)

    print(f"for 3 species, z = {z:.3f}")

    plt.figure(figsize=(6, 4))
    plt.plot(logL, logtau, "o", label="monte carlo data")
    plt.plot(logL, fit, "--", label=fr"$z \approx {z:.3f}$")
    
    plt.xlabel(r"$\log L$")
    plt.ylabel(r"$\log \tau(L)$")
    plt.title(r"dynamic critical exponent monte carlo")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("figures/dynamical_critical_exponent_simulation.png", dpi=300)
    plt.show()

    """
    solving for the critical dynamical exponent for n = 2, 3, 4, 5, 6, 7, 8, and 9 species
    """
    fig, axes = plt.subplots(4, 2, figsize=(10, 15)) 
    fig.suptitle("the critical dynamical exponent for many species")

    for species_size, ax in zip(np.arange(2, 10, 1), axes.flatten()): 
        logL, logtau, fit, z = get_dynamical_critical_exponent(species_size=species_size) 
        
        print(f"for {species_size} species, z = {z:.3f}")
        
        ax.plot(logL, logtau, "o", label="monte carlo data")
        ax.plot(logL, fit, "--", label=fr"$z \approx {z:.3f}$")
        ax.set_xlabel(r"$\log L$")
        ax.set_ylabel(r"$\log \tau(L)$")
        
        ax.set_title(f"species size = {species_size}")
        ax.legend()
        ax.grid(True)

    fig.tight_layout() 

    plt.savefig("figures/many_species_dynamical_critical_exponent_simulation.png", dpi=300)
    plt.show()

    