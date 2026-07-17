"""
running processes in serial demo
"""

# ./run.sh 1G 30:00 1 ../src/serial_msep.py

import numpy as np
from src.utils import MultiSpeciesExclusionProcess, autocorrelation, relaxation_time

def getzs(rates_matrix):
    tau1s = []
    tau2s = []
    L_values = np.arange(30, 300, 3)

    for L in L_values:
        msep_process = MultiSpeciesExclusionProcess(dimension=3, density=[1/3, 1/3, 1/3], length=L, shuffle=True, rates_matrix=rates_matrix, seed=L)
        X = msep_process.fourier_time_series()
        C1 = autocorrelation(X[:, 0])
        C2 = autocorrelation(X[:, 1])

        tau1s.append(relaxation_time(C1))
        tau2s.append(relaxation_time(C2))

    log_L = np.log(L_values)
    z1, _ = np.polyfit(log_L, np.log(tau1s), 1)
    z2, _ = np.polyfit(log_L, np.log(tau2s), 1)
    return z1, z2

if __name__ == "__main__":
    matrices = [
        np.array([[0.0, 0.1, 1.1], [2.1, 0.0, 3.1], [0.1, 0.1, 0.0]]),
        np.array([[0.0, 1.0, 0.1], [1.0, 0.0, 0.1], [2.1, 2.1, 0.0]]),
        np.array([[0.0, 0.1, 1.0], [2.1, 0.0, 2.1], [1.0, 0.1, 0.0]]),
        np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    ]

    print("Running MSEP simulation sequentially...")

    results = [getzs(m) for m in matrices]
    for idx, (z1, z2) in enumerate(results):
        print(f"Matrix {idx + 1} Exponents calculated: z1 = {z1:.4f}, z2 = {z2:.4f}")

    print(f"Done :)")