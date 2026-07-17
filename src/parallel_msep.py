"""
running processes in parallel demo
"""

# ./run.sh 4G 30:00 4 ../src/parallel_msep.py --start-ipyengines

import os
import numpy as np
import ipyparallel as ipp

def compute_matrix_regime(rates_matrix):
    import numpy as np
    from src.utils import MultiSpeciesExclusionProcess, autocorrelation, relaxation_time

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

    profile = os.environ.get("IPP_PROFILE", "default")
    
    rc = ipp.Client(profile=profile)
    view = rc.load_balanced_view()

    print(f"Distributing 4 matrix jobs across {len(rc.ids)} SLURM cluster engines...")
    
    results = view.map_sync(compute_matrix_regime, matrices)
    for idx, (z1, z2) in enumerate(results):
        print(f"Matrix {idx + 1} Exponents calculated: z1 = {z1:.4f}, z2 = {z2:.4f}")
    
    print(f"Done :)")
    rc.close()