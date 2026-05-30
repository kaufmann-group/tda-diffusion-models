import numpy as np
import matplotlib.pyplot as plt

from utils.autocorrelation import autocorrelation
from utils.msep import MultiSpeciesExclusionProcess
from utils.relaxation_time import relaxation_time


def getzs(rates_matrix):
    tau1s = []
    tau2s = []
    L_values = np.arange(30, 300, 3)

    for L in L_values:
        msep_process = MultiSpeciesExclusionProcess(dimension = 3, density = [1/3, 1/3, 1/3], length = L, shuffle = True, rates_matrix = rates_matrix)
        X = msep_process.fourier_time_series()
        C1 = autocorrelation(X[:, 0])
        C2 = autocorrelation(X[:, 1])

        t1 = relaxation_time(C1)
        t2 = relaxation_time(C2)

        tau1s.append(t1)
        tau2s.append(t2)

    log_L = np.log(L_values)
    log_t1 = np.log(tau1s)
    log_t2 = np.log(tau2s)


    z1, intercept1 = np.polyfit(log_L, log_t1, 1)
    z2, intercept2 = np.polyfit(log_L, log_t2, 1)

    fit1 = z1*log_L + intercept1
    fit2 = z2*log_L + intercept2

    return log_L, log_t1, log_t2, z1, z2, intercept1, intercept2, fit1, fit2

def plot(ax, rates_matrix):
    
    log_L, log_t1, log_t2, z1, z2, intercept1, intercept2, fit1, fit2 = getzs(rates_matrix)

    ax.plot(log_L, log_t1, "go", alpha = 0.5, ms = 4, label = f"z1 = {z1:.2f}")
    ax.plot(log_L, log_t2, "go", alpha = 0.5, ms = 4, label = f"z2 = {z2:.2f}")
    ax.plot(log_L, fit1, "b--", label = "Fit 1")
    ax.plot(log_L, fit2, "b--", label = "Fit 2")
    ax.set_xlabel("log(L)")
    ax.set_ylabel("log(τ)")
    ax.legend(fontsize=8)
    
if __name__ == "__main__":
    fig, axis = plt.subplots(2, 2, figsize = (8,8))
    
    rates_matrix_1 = np.array([
        [0.0, 0.1, 1.1],
        [2.1, 0.0, 3.1],
        [0.1, 0.1, 0.0]
    ])

    rates_matrix_2 = np.array([
        [0.0, 1.0, 0.1],
        [1.0, 0.0, 0.1],
        [2.1, 2.1, 0.0]
    ])

    rates_matrix_3 = np.array([
        [0.0, 0.1, 1.0],
        [2.1, 0.0, 2.1],
        [1.0, 0.1, 0.0]
    ])

    rates_matrix_4 = np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ])

    plot(ax = axis[0,0], rates_matrix = rates_matrix_1)
    plot(ax = axis[0,1], rates_matrix = rates_matrix_2)
    plot(ax = axis[1,0], rates_matrix = rates_matrix_3)
    plot(ax = axis[1,1], rates_matrix = rates_matrix_4)

    fig.suptitle("Relaxation Time Scaling for Different Rates Matrices", fontsize = 14)
    plt.tight_layout() 
    

    plt.savefig("../figures/timescaling.png", dpi=300)
    plt.show()

