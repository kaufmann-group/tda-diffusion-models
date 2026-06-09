import numpy as np

def widths(H):
    mean_H = H.mean(axis=1, keepdims=True)
    W = np.sqrt(np.mean((H - mean_H)**2, axis=1))
    return W