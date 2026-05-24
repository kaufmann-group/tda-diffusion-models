import numpy as np
import gudhi as gd


"""
computes tda observables from one 2d projected path snapshot ... 
"""

def tda_observables(point_cloud, r=1.5, epsilon=0.2, max_edge_length=5.0):
    # removes duplicate points ... 
    point_cloud = np.unique(point_cloud, axis=0)

    if point_cloud.shape[0] < 3:
        print("went :)")
        return 0, 0.0, 0.0, 0

    rips = gd.RipsComplex(points=point_cloud, max_edge_length=max_edge_length)

    # max_dimension=2 so h1 loops can die by being filled in according to ai ... idk what this means 
    simplex_tree = rips.create_simplex_tree(max_dimension=2)

    simplex_tree.persistence()

    h1 = simplex_tree.persistence_intervals_in_dimension(1)

    if h1.shape[0] > 0:
        h1 = h1[np.isfinite(h1[:, 1])]

    if h1.shape[0] == 0:
        return 0, 0.0, 0.0, 0

    births = h1[:, 0]
    deaths = h1[:, 1]
    persistences = deaths - births

    # beta_1(r,t): number of h1 bars alive at filtration scale r ... also don't know what this means
    beta_1 = np.sum((births <= r) & (r < deaths))

    # total h1 persistence
    p_total = np.sum(persistences)

    # max h1 persistence
    p_max = np.max(persistences)

    # number of loops with persistence above threshold epsilon ... why is this important
    n_epsilon = np.sum(persistences > epsilon)

    return int(beta_1), float(p_max), float(p_total), int(n_epsilon)