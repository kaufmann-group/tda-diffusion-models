"""
autocorrelation with fast fourier transform.
"""

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