import numpy as np

"""
gets the relaxation time.
"""

def relaxation_time(C):
    C = np.asarray(np.abs(C))

    if len(C) < 2:
        return np.nan

    # Normalize so C(0) = 1
    if C[0] <= 0 or not np.isfinite(C[0]):
        return np.nan

    C = C / C[0]

    threshold = np.exp(-1)

    crossings = np.where((np.arange(len(C)) > 0) & (C < threshold))[0]

    if len(crossings) == 0:
        return np.nan

    i = crossings[0]

    t0, t1 = i - 1, i
    C0, C1 = C[t0], C[t1]

    if C0 == C1:
        return float(i)

    tau = t0 + (threshold - C0) / (C1 - C0)

    return float(tau)