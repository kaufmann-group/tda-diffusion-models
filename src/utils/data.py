import numpy as np
from scipy.signal import correlate, savgol_filter

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

"""
get the saturation time
"""
def saturation_time(x, y):
    dy = np.gradient(y, x)

    window_len = 50
    dy_smooth = np.convolve(dy, np.ones(window_len)/window_len, mode='same')

    threshold = 0.10 * dy_smooth[0]
    saturation_idx = np.where(dy_smooth < threshold)[0][0]

    saturation_time = x[saturation_idx]
    saturation_value = y[saturation_idx]

    return x[saturation_idx], y[saturation_idx]

"""
smoothen with Savgol filter
"""
def smooth(y):
    window_length = 1001
    polyorder = 3

    return savgol_filter(y, window_length=window_length, polyorder=polyorder)

"""
calculate autocorrelation for time series
"""
def autocorrelation(x, max_lag=None):
    x = np.asarray(x, dtype=np.complex128).copy()

    n = len(x)

    if n == 0:
        return np.array([])

    x -= np.mean(x)

    raw = correlate(x, x, mode="full", method="fft")

    C = raw[n - 1:]
    C = C / n

    if not np.isfinite(C[0]) or np.abs(C[0]) == 0:
        return np.full(n, np.nan, dtype=np.complex128)

    C = C / C[0]

    if max_lag is not None:
        C = C[:max_lag]

    return C

"""
fit the dynamical critical exponent
"""

def fit_dynamic_exponent(L_values, saturation_times):
    L_values = np.asarray(L_values, dtype=float)
    saturation_times = np.asarray(saturation_times, dtype=float)

    valid = np.isfinite(saturation_times) & (saturation_times > 0)

    log_L = np.log(L_values[valid])
    log_tau = np.log(saturation_times[valid])

    z, intercept = np.polyfit(log_L, log_tau, 1)

    return z, intercept, log_L, log_tau
