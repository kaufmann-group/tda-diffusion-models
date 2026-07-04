import numpy as np
import gudhi as gd
from ripser import ripser
from matplotlib.patches import Polygon

# Default observable parameters

BETA_R_H0 = 1.0
EPS_H1 = 0.05

R_MIN = 0.0
R_MAX = 3.0
N_R_GRID = 80
R_GRID = np.linspace(R_MIN, R_MAX, N_R_GRID)

# Internal helpers

"""
Return points as a finite 2D float array.
"""
def _as_point_array(points):
    points = np.asarray(points, dtype=float)

    if points.ndim != 2:
        return np.empty((0, 0), dtype=float)

    if points.shape[0] == 0 or points.shape[1] == 0:
        return np.empty((0, points.shape[1] if points.ndim == 2 else 0), dtype=float)

    finite_rows = np.all(np.isfinite(points), axis=1)
    return points[finite_rows]

"""
Compute ripser persistence diagrams after basic point cloud validation.
"""
def _ripser_diagrams(points, maxdim):
    if gd is None:
        raise ImportError("draw_rips_simplices requires gudhi to be installed")

    points = _as_point_array(points)

    if points.shape[0] == 0:
        empty = np.empty((0, 2), dtype=float)
        if maxdim <= 0:
            return [empty]
        return [empty for _ in range(maxdim + 1)]

    return ripser(points, maxdim=maxdim)["dgms"]

"""
Return the H1 diagram of a point cloud, or an empty diagram if unavailable.
"""
def _h1_diagram(points):
    points = _as_point_array(points)

    if points.shape[0] < 3:
        return np.empty((0, 2), dtype=float)

    diagrams = _ripser_diagrams(points, maxdim=1)

    if len(diagrams) < 2:
        return np.empty((0, 2), dtype=float)

    return diagrams[1]

# Persistence utilities

"""
Return finite positive persistence lifetimes death minus birth.
"""
def finite_lifetimes(diagram):
    if diagram is None or len(diagram) == 0:
        return np.asarray([], dtype=float)

    diagram = np.asarray(diagram, dtype=float)

    if diagram.ndim != 2 or diagram.shape[1] < 2:
        return np.asarray([], dtype=float)

    births = diagram[:, 0]
    deaths = diagram[:, 1]

    finite = np.isfinite(births) & np.isfinite(deaths)
    lifetimes = deaths[finite] - births[finite]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    lifetimes = lifetimes[lifetimes > 0]

    return lifetimes
    
"""
Return only finite birth death pairs with positive persistence.
"""
def finite_diagram(diagram):
    if diagram is None or len(diagram) == 0:
        return np.empty((0, 2), dtype=float)

    diagram = np.asarray(diagram, dtype=float)

    if diagram.ndim != 2 or diagram.shape[1] < 2:
        return np.empty((0, 2), dtype=float)

    finite = np.isfinite(diagram[:, 0]) & np.isfinite(diagram[:, 1])
    diagram = diagram[finite]

    if len(diagram) == 0:
        return np.empty((0, 2), dtype=float)

    lifetimes = diagram[:, 1] - diagram[:, 0]
    diagram = diagram[lifetimes > 0]

    return diagram


# ============================================================
# Projection-path / GUDHI observables
# ============================================================

"""
Compute beta_1(r), the number of H1 loops alive at filtration scale r.
"""
def beta_1(point_cloud, r=1.5, max_edge_length=5.0):
    if gd is None:
        raise ImportError("beta_1 requires gudhi to be installed")

    if gd is None:
        raise ImportError("p_max requires gudhi to be installed")

    point_cloud = _as_point_array(point_cloud)

    if point_cloud.shape[0] < 3:
        return 0

    point_cloud = np.unique(point_cloud, axis=0)

    if point_cloud.shape[0] < 3:
        return 0

    rips = gd.RipsComplex(points=point_cloud, max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    simplex_tree.persistence()

    h1 = simplex_tree.persistence_intervals_in_dimension(1)

    if h1.shape[0] == 0:
        return 0

    h1 = h1[np.isfinite(h1[:, 1])]

    if h1.shape[0] == 0:
        return 0

    births = h1[:, 0]
    deaths = h1[:, 1]

    return int(np.sum((births <= r) & (r < deaths)))

"""
Compute the largest finite H1 persistence lifetime in a projected point cloud.
"""
def p_max(point_cloud, max_edge_length=5.0):
    point_cloud = _as_point_array(point_cloud)

    if point_cloud.shape[0] < 3:
        return 0.0

    point_cloud = np.unique(point_cloud, axis=0)

    if point_cloud.shape[0] < 3:
        return 0.0

    rips = gd.RipsComplex(points=point_cloud, max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    simplex_tree.persistence()

    h1 = simplex_tree.persistence_intervals_in_dimension(1)

    if h1.shape[0] == 0:
        return 0.0

    h1 = h1[np.isfinite(h1[:, 1])]

    if h1.shape[0] == 0:
        return 0.0

    lifetimes = h1[:, 1] - h1[:, 0]
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return 0.0

    return float(np.max(lifetimes))

"""
Draw vertices, edges, and triangles of a Vietoris Rips complex at scale d
"""
def draw_rips_simplices(points, axes, d, max_dimension=2, show_labels=False):
    points = _as_point_array(points)

    if points.shape[0] == 0:
        axes.set_title(f"rips complex simplices for d = {d}")
        axes.set_aspect("equal")
        return

    rips = gd.RipsComplex(points=points, max_edge_length=d)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension)

    edges = []
    triangles = []

    for simplex, filt in simplex_tree.get_filtration():
        if filt <= d:
            dim = len(simplex) - 1
            if dim == 1:
                edges.append(simplex)
            elif dim == 2:
                triangles.append(simplex)

    for tri in triangles:
        coords = points[list(tri)]
        poly = Polygon(coords, closed=True, facecolor="purple", alpha=0.25, edgecolor=None)
        axes.add_patch(poly)

    for i, j in edges:
        axes.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color="black", linewidth=2)

    axes.scatter(points[:, 0], points[:, 1], s=180, color="steelblue", edgecolor="black", zorder=3)

    if show_labels:
        for i, (x, y) in enumerate(points):
            axes.text(x + 0.05, y + 0.05, str(i), fontsize=10)

    axes.set_title(f"rips complex simplices for d = {d}")
    axes.set_aspect("equal")

# Patch construction

"""
Convert a 1D height profile into local sliding window patches.
"""
def patch_point_cloud(h_profile, window, stride=1):
    h = np.asarray(h_profile, dtype=float)
    h = h[np.isfinite(h)]

    window = int(window)
    stride = int(stride)

    if window <= 0:
        raise ValueError("window must be positive")

    if stride <= 0:
        raise ValueError("stride must be positive")

    if len(h) < window:
        return np.empty((0, window), dtype=float)

    h = h - np.mean(h)

    std = np.std(h)
    if std > 1e-14:
        h = h / std

    points = [
        h[start:start + window]
        for start in range(0, len(h) - window + 1, stride)
    ]

    return np.asarray(points, dtype=float)

# H0 point cloud observables

"""
Compute total finite H0 persistence of a point cloud.
"""
def h0_total_persistence_from_points(points):
    diagrams = _ripser_diagrams(points, maxdim=0)
    lifetimes = finite_lifetimes(diagrams[0])

    if len(lifetimes) == 0:
        return 0.0

    return float(np.sum(lifetimes))

"""
Compute the maximum finite H0 persistence lifetime of a point cloud.
"""
def h0_max_persistence_from_points(points):
    diagrams = _ripser_diagrams(points, maxdim=0)
    lifetimes = finite_lifetimes(diagrams[0])

    if len(lifetimes) == 0:
        return 0.0

    return float(np.max(lifetimes))

"""
Compute beta_0(r), the number of H0 classes alive at filtration scale r.
"""
def h0_beta_fixed_r_from_points(points, r=BETA_R_H0):
    diagrams = _ripser_diagrams(points, maxdim=0)
    dgm0 = diagrams[0]

    if dgm0 is None or len(dgm0) == 0:
        return 0.0

    births = dgm0[:, 0]
    deaths = dgm0[:, 1]
    alive = (births <= r) & (r < deaths)

    return float(np.sum(alive))

"""
Compute persistent entropy of finite H0 lifetimes.
"""
def h0_entropy_from_points(points):
    diagrams = _ripser_diagrams(points, maxdim=0)
    lifetimes = finite_lifetimes(diagrams[0])

    if len(lifetimes) == 0:
        return 0.0

    total = np.sum(lifetimes)

    if total <= 0:
        return 0.0

    probabilities = lifetimes / total
    probabilities = probabilities[probabilities > 0]

    return float(-np.sum(probabilities * np.log(probabilities)))

# H1 point cloud observables

"""
Compute total finite H1 persistence of a point cloud.
"""
def h1_total_persistence_from_points(points):
    dgm1 = _h1_diagram(points)
    lifetimes = finite_lifetimes(dgm1)

    if len(lifetimes) == 0:
        return 0.0

    return float(np.sum(lifetimes))

"""
Compute the maximum finite H1 persistence lifetime of a point cloud.
"""
def h1_max_persistence_from_points(points):
    dgm1 = _h1_diagram(points)
    lifetimes = finite_lifetimes(dgm1)

    if len(lifetimes) == 0:
        return 0.0

    return float(np.max(lifetimes))

"""
Count H1 features whose finite persistence lifetime is larger than eps.
"""
def h1_num_persistent_from_points(points, eps=EPS_H1):
    dgm1 = _h1_diagram(points)
    lifetimes = finite_lifetimes(dgm1)

    if len(lifetimes) == 0:
        return 0.0

    return float(np.sum(lifetimes > eps))

"""
Compute persistent entropy of finite H1 lifetimes.
"""
def h1_entropy_from_points(points):
    dgm1 = _h1_diagram(points)
    lifetimes = finite_lifetimes(dgm1)

    if len(lifetimes) == 0:
        return 0.0

    total = np.sum(lifetimes)

    if total <= 0:
        return 0.0

    probabilities = lifetimes / total
    probabilities = probabilities[probabilities > 0]

    return float(-np.sum(probabilities * np.log(probabilities)))

# Beta curve / CROCKER observables
    
"""
Compute the Betti curve beta_k(r) over a filtration grid.
"""
def betti_curve(diagram, r_grid=R_GRID):
    if diagram is None or len(diagram) == 0:
        return np.zeros(len(r_grid), dtype=float)

    diagram = np.asarray(diagram, dtype=float)

    if diagram.ndim != 2 or diagram.shape[1] < 2:
        return np.zeros(len(r_grid), dtype=float)

    births = diagram[:, 0]
    deaths = diagram[:, 1]

    valid = np.isfinite(births) & (np.isfinite(deaths) | np.isinf(deaths))
    births = births[valid]
    deaths = deaths[valid]

    curve = np.zeros(len(r_grid), dtype=float)

    for i, r in enumerate(r_grid):
        curve[i] = np.sum((births <= r) & (r < deaths))

    return curve

"""
Compute the area under a Betti curve.
"""
def beta_curve_area_from_diagram(diagram):
    curve = betti_curve(diagram, R_GRID)
    return float(np.trapz(curve, R_GRID))

"""
Compute the L2 norm of a Betti curve, also called a CROCKER L2 norm here.
"""
def beta_curve_l2_norm_from_diagram(diagram):
    curve = betti_curve(diagram, R_GRID)
    return float(np.sqrt(np.trapz(curve ** 2, R_GRID)))

"""
Compute the area under the H0 Betti curve of a point cloud.
"""
def h0_beta_curve_area_from_points(points):
    diagrams = _ripser_diagrams(points, maxdim=0)
    return beta_curve_area_from_diagram(diagrams[0])

"""
Compute the area under the H1 Betti curve of a point cloud.
"""
def h1_beta_curve_area_from_points(points):
    dgm1 = _h1_diagram(points)
    return beta_curve_area_from_diagram(dgm1)

"""
Compute the H0 CROCKER L2 norm of a point cloud.
"""
def h0_crocker_l2_norm_from_points(points):
    diagrams = _ripser_diagrams(points, maxdim=0)
    return beta_curve_l2_norm_from_diagram(diagrams[0])

"""
Compute the H1 CROCKER L2 norm of a point cloud.
"""
def h1_crocker_l2_norm_from_points(points):
    dgm1 = _h1_diagram(points)
    return beta_curve_l2_norm_from_diagram(dgm1)