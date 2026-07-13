import os
import signal
import numpy as np


def safe_name(name):
    return str(name).replace("/", "_").replace(" ", "_").replace(":", "_")

"""
Simple checkpointing for jobs indexed by L.

Saves to:
    data/slurm-jobs/<process_name>.npz

Stores:
    results[L_index, output_index]
    done[L_index]
"""
class LCheckpoint:

    def __init__(self, process_name, L_values, n_outputs=2, output_dir="data/slurm-jobs"):
        self.process_name = safe_name(process_name)
        self.L_values = np.asarray(L_values, dtype=int)
        self.n_outputs = int(n_outputs)

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.filename = os.path.join(self.output_dir, self.process_name + ".npz")

        self.results = np.full((len(self.L_values), self.n_outputs), np.nan, dtype=float)
        self.done = np.zeros(len(self.L_values), dtype=bool)

        self._load_if_exists()

    def _load_if_exists(self):
        if not os.path.exists(self.filename):
            print("No checkpoint found. Starting fresh:")
            print("  ", self.filename)
            return

        data = np.load(self.filename, allow_pickle=True)

        old_L_values = data["L_values"]
        old_n_outputs = int(data["n_outputs"])

        if not np.array_equal(old_L_values, self.L_values):
            raise ValueError("Checkpoint L_values do not match current L_values. Use a new process_name or delete the old checkpoint.")

        if old_n_outputs != self.n_outputs:
            raise ValueError("Checkpoint n_outputs does not match current n_outputs. Use a new process_name or delete the old checkpoint.")

        self.results = data["results"]
        self.done = data["done"]

        print("Loaded checkpoint:")
        print("  ", self.filename)
        print("Completed L jobs:", int(np.sum(self.done)), "/", len(self.done))

    def save(self):
        tmp_filename = self.filename + ".tmp.npz"

        np.savez(tmp_filename, process_name=self.process_name, L_values=self.L_values, n_outputs=self.n_outputs, results=self.results, done=self.done)

        os.replace(tmp_filename, self.filename)

    def L_to_index(self, L):
        matches = np.where(self.L_values == int(L))[0]

        if len(matches) == 0:
            raise ValueError(f"L={L} is not in L_values")

        return int(matches[0])

    def record(self, L, values, autosave=True):
        L_index = self.L_to_index(L)

        values = np.asarray(values, dtype=float)

        if values.shape != (self.n_outputs,):
            raise ValueError(f"Expected values shape {(self.n_outputs,)}, got {values.shape}")

        self.results[L_index, :] = values
        self.done[L_index] = True

        if autosave:
            self.save()

    def remaining_L_values(self):
        return [int(L) for L_index, L in enumerate(self.L_values) if not self.done[L_index]]

    def get_results(self):
        return self.results.copy()

    def print_status(self):
        print(f"{self.process_name}: {int(np.sum(self.done))}/{len(self.done)} L jobs complete")

    def install_signal_handlers(self):
        def handler(signum, frame):
            print()
            print(f"Received signal {signum}. Saving checkpoint...")
            self.save()
            print("Checkpoint saved:")
            print("  ", self.filename)
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    """
    Delete this checkpoint file.
    """
    def delete(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)
            print("Deleted checkpoint:")
            print("  ", self.filename)
        else:
            print("No checkpoint file to delete:")
            print("  ", self.filename)

    """
    Return True if all L jobs are complete.
    """
    def is_complete(self):
        return bool(np.all(self.done))