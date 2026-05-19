from msep.msep_cpp import MultiSpeciesExclusionProcess
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt

"""
the goal of this program is to implement TDA on point cloud x
of multi-species exclusion processes. 
"""

def persistence(point_cloud, epsilon=0.2, max_edge_length=5.0):
    points = np.asarray(point_cloud, dtype=float)

    points = np.unique(points, axis=0)

    rips = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)

    simplex_tree.persistence()

    h1 = simplex_tree.persistence_intervals_in_dimension(1)

    if h1.shape[0] == 0:
        return 0.0, 0.0, 0

    h1 = h1[np.isfinite(h1[:, 1])]

    if h1.shape[0] == 0:
        return 0.0, 0.0, 0

    h1_persistences = h1[:, 1] - h1[:, 0]

    p_max = np.max(h1_persistences)
    p_total = np.sum(h1_persistences)
    n_epsilon = np.sum(h1_persistences > epsilon)

    return p_max, p_total, n_epsilon


if __name__ == "__main__":
    """
    persistence diagram from 2D point cloud from two species 
    totally asymmetric simple exclusion process.
    """

    rates_matrix = np.array(
        [
            [0.0, 2.0, 2.0],
            [1.0, 0.0, 1.5],
            [1.0, 1.5, 0.0],
        ],
        dtype=np.float64,
    )

    model = MultiSpeciesExclusionProcess(dimension=3, density=[1/3,1/3,1/3], rates_matrix=rates_matrix, length=45, seed=2504)
    path_history = model.simulate(steps=100000, store_history=True, get_projection=True)

    p_maxs, p_totals, n_epsilons = map(list, zip(*(persistence(path) for path in path_history)))
    
    plt.plot(p_maxs, label="max H1 persistence")
    plt.plot(p_totals, label="total H1 persistence")
    plt.plot(n_epsilons, label="number of persistent H1 loops")
    plt.legend()

    plt.savefig("figures/persistance_diagrams.png", dpi=300)
    plt.show()
