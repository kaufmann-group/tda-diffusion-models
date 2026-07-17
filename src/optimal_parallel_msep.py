import os
import numpy as np
import ipyparallel as ipp

# ./run.sh 1G 30:00 25 ../src/optimal_parallel_msep.py --start-ipyengines

# handles one simulation
def compute_single_simulation(task_args):
    import numpy as np
    from src.utils import MultiSpeciesExclusionProcess, autocorrelation, relaxation_time

    matrix_idx, rates_matrix, L = task_args

    msep_process = MultiSpeciesExclusionProcess(dimension=3, density=[1/3, 1/3, 1/3], length=L, shuffle=True, rates_matrix=rates_matrix, seed=L)
    X = msep_process.fourier_time_series()

    t1 = relaxation_time(autocorrelation(X[:, 0]))
    t2 = relaxation_time(autocorrelation(X[:, 1]))

    # return metadata so the main thread knows how to piece the results back together
    return matrix_idx, L, t1, t2


if __name__ == "__main__":
    matrices = [
        np.array([[0.0, 0.1, 1.1], [2.1, 0.0, 3.1], [0.1, 0.1, 0.0]]),
        np.array([[0.0, 1.0, 0.1], [1.0, 0.0, 0.1], [2.1, 2.1, 0.0]]),
        np.array([[0.0, 0.1, 1.0], [2.1, 0.0, 2.1], [1.0, 0.1, 0.0]]),
        np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    ]
    
    L_values = np.arange(30, 300, 3) # 90 values

    # flatten nested loops into a single list of 360 independent tasks
    tasks = []
    for m_idx, matrix in enumerate(matrices):
        for L in L_values:
            tasks.append((m_idx, matrix, L))

    profile = os.environ.get("IPP_PROFILE", "default")
    rc = ipp.Client(profile=profile)
    view = rc.load_balanced_view()

    print(f"Distributing {len(tasks)} flattened simulation tasks across {len(rc.ids)} SLURM engines...")
    
    # fire all 360 runs out to engines concurrently
    raw_results = view.map_sync(compute_single_simulation, tasks)
    rc.close()

    print("All tasks gathered! Reassembling and calculating scaling exponents...")

    # group unordered parallel results back by their original matrix index
    grouped_data = {i: {"L": [], "tau1": [], "tau2": []} for i in range(len(matrices))}
    
    for matrix_idx, L, t1, t2 in raw_results:
        grouped_data[matrix_idx]["L"].append(L)
        grouped_data[matrix_idx]["tau1"].append(t1)
        grouped_data[matrix_idx]["tau2"].append(t2)

    # compute the final dce (z1, z2) for each regime
    final_results = []
    for m_idx in range(len(matrices)):
        # sort by L just in case engines returned out of chronological order
        sort_idx = np.argsort(grouped_data[m_idx]["L"])
        sorted_L = np.array(grouped_data[m_idx]["L"])[sort_idx]
        sorted_t1 = np.array(grouped_data[m_idx]["tau1"])[sort_idx]
        sorted_t2 = np.array(grouped_data[m_idx]["tau2"])[sort_idx]

        log_L = np.log(sorted_L)
        z1, _ = np.polyfit(log_L, np.log(sorted_t1), 1)
        z2, _ = np.polyfit(log_L, np.log(sorted_t2), 1)
        
        final_results.append((z1, z2))
        print(f"Matrix {m_idx + 1} Exponents calculated: z1 = {z1:.4f}, z2 = {z2:.4f}")

    print(f"Done :)")