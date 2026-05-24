"""
autocorrelation with fast fourier transform, the old autocorrelation 
code is below that uses unbiased normalization. 

import numpy as np
import scipy as sp

def autocorrelation(x):
    signal = np.asarray(x)
    signal -= np.mean(signal)

    n_samples = len(signal)

    raw_ac = sp.signal.correlate(signal, signal, mode="full", method="fft")
    positive_lags_ac = raw_ac[n_samples - 1 :]

    # unbiased normalization
    unbiased_ac = positive_lags_ac / np.arange(n_samples, 0, -1)
    
    return unbiased_ac / unbiased_ac[0]
"""

import numpy as np
from scipy.signal import correlate

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