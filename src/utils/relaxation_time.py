import numpy as np

def get_relaxation_time(C, L):
    t = np.arange(len(C))
    envelope = np.abs(C)

    # smoothing to reduce monte carlo noise
    window = 7
    kernel = np.ones(window) / window
    envelope_smooth = np.convolve(envelope, kernel, mode="same")

    # time where envelope has decayed significantly
    crossings = np.where((t > 0) & (envelope_smooth < 0.2))[0]

    if len(crossings) > 0:
        end = crossings[0]
    else:
        end = int(min(len(t), 4 * L ** 1.5))

    # fir where decay is neither too early nor too noisy
    mask = ((t > 0) & (np.arange(len(t)) < end) & (envelope_smooth < 0.85) & (envelope_smooth > 0.25))

    if np.sum(mask) < 8:
        mask = ((t > 0) & (np.arange(len(t)) < end) & (envelope_smooth < 0.9) & (envelope_smooth > 0.15))

    slope, _ = np.polyfit(t[mask], np.log(envelope_smooth[mask]), 1)
    
    # this part is sketchy ... 
    if slope >= 0: 
        return np.nan
    else:
        return -1.0 / slope