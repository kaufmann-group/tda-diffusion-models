"""
Checkpointed two-regime DCE test using h1_total_persistence_from_points.
"""

from .dce_two_regime import run

LOAD_PREVIOUS_SLURM_JOB = True

if __name__ == "__main__":
    observable_name = "h1_total_persistence_from_points"
    process_name = "raw_projection_h1_total_persistence_two_regimes"
    output_filename = "h1_total_persistence_two_regimes.png"
    suptitle = r"$H_1$ Total Persistence Relaxation Time Scaling"
    load_previous_slurm_job = LOAD_PREVIOUS_SLURM_JOB

    run(observable_name=observable_name, process_name=process_name, output_filename=output_filename, suptitle=suptitle, load_previous_slurm_job=load_previous_slurm_job)

