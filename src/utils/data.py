import numpy as np
from scipy.signal import correlate, savgol_filter

"""
Calculates the relaxation time when C(t) crosses below a threshold.
"""

import numpy as np

def relaxation_time(C, times=None, threshold=None):
    C = np.asarray(C) 
    
    if np.iscomplexobj(C):             
        C = np.abs(C)
        
    C = C.astype(float)

    C = np.asarray(C, dtype=float)
    
    if times is None:
        C = np.abs(C)
        if len(C) < 2 or C[0] <= 0 or not np.isfinite(C[0]):
            return np.nan
        C = C / C[0]
        times = np.arange(len(C), dtype=float)
    else:
        times = np.asarray(times, dtype=float)

    finite = np.isfinite(C) & np.isfinite(times)
    C = C[finite]
    times = times[finite]

    if len(C) < 2:
        return np.nan

    if threshold is None:
        threshold = np.exp(-1)

    below = np.where(C < threshold)[0]
    if len(below) == 0:
        return np.nan

    i = below[0]

    if i == 0:
        return float(times[0])

    t0, t1 = times[i - 1], times[i]
    c0, c1 = C[i - 1], C[i]

    if np.isclose(c1, c0, atol=1e-14):
        return float(t1)

    tau = t0 + (threshold - c0) * (t1 - t0) / (c1 - c0)
    return float(tau)



"""
get the saturation time for noisy data

Saturation value is estimated from the final part of the curve.
Saturation time is the first time where the curve stays close to that value
for a full window.
"""
def saturation_time(x, y, steady_fraction=0.2, tolerance_fraction=0.10, window_fraction=0.05):
    x = np.asarray(x)
    y = np.asarray(y)

    n = len(y)

    steady_start = int((1.0 - steady_fraction) * n)
    steady_region = y[steady_start:]
    saturation_value = np.mean(steady_region)

    tolerance = tolerance_fraction * abs(saturation_value)
    if tolerance == 0:
        tolerance = tolerance_fraction

    window_len = max(1, int(window_fraction * n))

    for i in range(0, n - window_len):
        window = y[i:i + window_len]

        if np.all(np.abs(window - saturation_value) <= tolerance):
            return x[i], saturation_value

    return np.nan, saturation_value

"""
smoothen with Savgol filter
"""
def smooth(y, polyorder = 3, window_length = 101):

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
fit the dynamical critical exponent delete this 
"""
def fit_dynamic_exponent(L_values, saturation_times):
    L_values = np.asarray(L_values, dtype=float)
    saturation_times = np.asarray(saturation_times, dtype=float)

    valid = np.isfinite(saturation_times) & (saturation_times > 0)

    log_L = np.log(L_values[valid])
    log_tau = np.log(saturation_times[valid])

    z, intercept = np.polyfit(log_L, log_tau, 1)

    return z, intercept, log_L, log_tau

def fit_loglog(L_values, taus):
    """Fit tau proportional to L to the power z."""
    L_values = np.asarray(L_values, dtype=float)
    taus = np.asarray(taus, dtype=float)

    valid = np.isfinite(taus) & (taus > 0)

    log_L = np.log(L_values[valid])
    log_tau = np.log(taus[valid])

    if len(log_L) < 2:
        return log_L, log_tau, np.nan, np.full_like(log_L, np.nan)

    z, intercept = np.polyfit(log_L, log_tau, 1)
    fit = z * log_L + intercept

    return log_L, log_tau, z, fit