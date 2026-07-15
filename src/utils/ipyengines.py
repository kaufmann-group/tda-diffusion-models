import os
import git_root
import numpy as np
import importlib
import ipyparallel as ipp

from src.utils import *

"""
Create one engine task for every rates-matrix and run-index combination at one system size.
"""
def make_tasks_for_L(L, observable_name, n_runs, n_samples, sample_every, patch_divisor, patch_stride, tda_every, module_path="src.hydro_mode_tda_dce.dce_four_regime"):
    tasks = []

    regime_module = importlib.import_module(module_path)
    RATES_MATRICES = regime_module.RATES_MATRICES

    for rates_index, rates_matrix in enumerate(RATES_MATRICES):
        for run_id in range(n_runs):
            tasks.append((int(L), rates_index, rates_matrix, run_id, observable_name, n_samples, sample_every, patch_divisor, patch_stride, tda_every))

    return tasks

"""
Connect to the existing ipyparallel controller and prepare all engines.
"""
def connect_to_engines(profile="default"):
    
    profile = os.environ.get("IPP_PROFILE", profile)
    rc = ipp.Client(profile=profile)

    if len(rc.ids) == 0:
        raise RuntimeError("No ipyparallel engines are connected.")

    repo_root = git_root.git_root()
    direct_view = rc[:]

    direct_view.execute(f"import os\nos.chdir({repo_root!r})", block=True)
    direct_view.execute("import numpy as np\nimport src.utils", block=True)

    direct_view.use_cloudpickle()

    load_balanced_view = rc.load_balanced_view()

    return rc, load_balanced_view

"""
Run the checkpointed four-regime calculation using one ipyparallel task per simulation run.
"""
def run(observable_name, process_name, output_filename, suptitle=None, load_previous_slurm_job=True, profile="default", rc=None, module_path="src.hydro_mode_tda_dce.dce_four_regime"):

    regime_module = importlib.import_module(module_path)
    
    single_run = regime_module.single_run
    RATES_MATRICES = regime_module.RATES_MATRICES
    aggregate_results = regime_module.aggregate_results
    plot_results = regime_module.plot_results

    L_values = np.arange(240, 600, 15)

    n_runs = 24
    n_samples = 6000
    sample_every = 25

    patch_divisor = 8
    patch_stride = 1
    tda_every = 2

    checkpoint_dir = f"{git_root.git_root()}/data/slurm_jobs"
    checkpoint = LCheckpoint(process_name=process_name, L_values=L_values, n_outputs=2 * len(RATES_MATRICES), output_dir=checkpoint_dir)

    if not load_previous_slurm_job:
        checkpoint.delete()
        checkpoint = LCheckpoint(process_name=process_name, L_values=L_values, n_outputs=2 * len(RATES_MATRICES), output_dir=checkpoint_dir)

    checkpoint.install_signal_handlers()

    if rc is None:
        rc, load_balanced_view = connect_to_engines(profile=profile)
    else:
        if len(rc.ids) == 0:
            raise RuntimeError("The provided ipyparallel client has no connected engines.")

        direct_view = rc[:]
        direct_view.use_cloudpickle()

        load_balanced_view = rc.load_balanced_view()

    n_engines = len(rc.ids)
    tasks_per_L = len(RATES_MATRICES) * n_runs

    print(f"observable = {observable_name}")
    print(f"process_name = {process_name}")
    print(f"checkpoint = {checkpoint.filename}")
    print(f"connected engines = {n_engines}")
    print(f"tasks submitted per L = {tasks_per_L}")

    if n_engines > tasks_per_L:
        print(f"Warning: there are {n_engines} engines but only {tasks_per_L} independent tasks per L.")

    checkpoint.print_status()

    for L in checkpoint.remaining_L_values():
        print()
        print(f"running L = {L}")

        tasks = make_tasks_for_L(L=L, observable_name=observable_name, n_runs=n_runs, n_samples=n_samples, sample_every=sample_every, patch_divisor=patch_divisor, patch_stride=patch_stride, tda_every=tda_every)

        print(f"  submitting {len(tasks)} individual runs to {n_engines} engines")

        async_result = load_balanced_view.map_async(single_run, tasks, ordered=False)
        async_result.wait_interactive()

        results = async_result.get()

        if len(results) != tasks_per_L:
            raise RuntimeError(f"L={L} returned {len(results)} tasks; expected {tasks_per_L}.")

        row_values = aggregate_results(results=results, n_runs=n_runs)

        checkpoint.record(L, row_values, autosave=True)
        checkpoint.print_status()

    tau_table = np.asarray(checkpoint.get_results(), dtype=float)

    plot_results(L_values=L_values, tau_table=tau_table, observable_name=observable_name, output_filename=output_filename, suptitle=suptitle)

    if checkpoint.is_complete():
        print("Checkpoint complete:")
        print("  ", checkpoint.filename)

    return L_values, tau_table