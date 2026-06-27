import numpy as np
import gudhi as gd
from ripser import ripser

from matplotlib.patches import Polygon



"""
computes beta_1 from one 2d projected path snapshot ... 
"""

def beta_1(point_cloud, r=1.5, max_edge_length=5.0):
    point_cloud = np.unique(point_cloud, axis=0)

    if point_cloud.shape[0] < 3:
        return 0

    rips = gd.RipsComplex(points=point_cloud, max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    simplex_tree.persistence()

    h1 = simplex_tree.persistence_intervals_in_dimension(1)

    if h1.shape[0] > 0:
        h1 = h1[np.isfinite(h1[:, 1])]

    if h1.shape[0] == 0:
        return 0

    births = h1[:, 0]
    deaths = h1[:, 1]

    beta_1_value = np.sum((births <= r) & (r < deaths))

    return int(beta_1_value)

"""
computes p_max from one 2d projected path snapshot

p_max is the largest H1 persistence value in the persistence diagram; for each H1 loop persistence = death - birth.
"""

def p_max(point_cloud, max_edge_length=5.0):
    point_cloud = np.unique(point_cloud, axis=0)

    if point_cloud.shape[0] < 3:
        return 0.0

    rips = gd.RipsComplex(points=point_cloud, max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    simplex_tree.persistence()

    h1 = simplex_tree.persistence_intervals_in_dimension(1)

    if h1.shape[0] > 0:
        h1 = h1[np.isfinite(h1[:, 1])]

    if h1.shape[0] == 0:
        return 0.0

    births = h1[:, 0]
    deaths = h1[:, 1]

    persistences = deaths - births
    p_max_value = np.max(persistences)

    return float(p_max_value)


"""
draw the rips simplices
"""
def draw_rips_simplices(points, axes, d, max_dimension=2, show_labels=False):
    rips = gd.RipsComplex(points=points, max_edge_length=d)
    st = rips.create_simplex_tree(max_dimension=max_dimension)

    vertices = []
    edges = []
    triangles = []

    for simplex, filt in st.get_filtration():
        if filt <= d:
            dim = len(simplex) - 1
            if dim == 0:
                vertices.append(simplex)
            elif dim == 1:
                edges.append(simplex)
            elif dim == 2:
                triangles.append(simplex)

    for tri in triangles:
        coords = points[list(tri)]
        poly = Polygon(coords, closed=True, facecolor='purple', alpha=0.25, edgecolor=None)
        axes.add_patch(poly)

    for e in edges:
        i, j = e
        axes.plot([points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
                color='black', linewidth=2)

    axes.scatter(points[:, 0], points[:, 1], s=180, color='steelblue',
               edgecolor='black', zorder=3)

    if show_labels:
        for i, (x, y) in enumerate(points):
            axes.text(x + 0.05, y + 0.05, str(i), fontsize=10)

    axes.set_title(f"rips complex simplices for d = {d}")
    axes.set_aspect('equal')

"""
Total finite H0 persistence of a point cloud.
"""

def h0_total_persistence(points):
    if points.ndim != 2 or points.shape[0] < 2:
        return 0.0

    result = ripser(points, maxdim=0)
    H0 = result["dgms"][0]

    births = H0[:, 0]
    deaths = H0[:, 1]

    finite = np.isfinite(births) & np.isfinite(deaths)

    if not np.any(finite):
        return 0.0

    pers = deaths[finite] - births[finite]

    return float(np.sum(pers))

    """
Convert 1D data into a point cloud of local spatial patches.
"""
def patch_point_cloud(h_profile, window, stride=1):
    
    h = np.asarray(h_profile, dtype=float)
    h = h[np.isfinite(h)]

    if len(h) < window:
        return np.empty((0, window))

    h = h - np.mean(h)

    std = np.std(h)
    if std > 1e-14:
        h = h / std

    points = []

    for start in range(0, len(h) - window + 1, stride):
        patch = h[start:start + window]
        points.append(patch)

    return np.asarray(points, dtype=float)