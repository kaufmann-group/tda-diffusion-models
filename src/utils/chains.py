"""
different chain configurations
"""

import numpy as np

def equal_spread_chain(L, d = 3):
    if L % d != 0:
        raise ValueError(f"needs to be divisible by {d} :) ")

    chain = np.zeros(L, dtype=np.int32)

    for j in range(L):
        chain[j] = j % d

    return chain