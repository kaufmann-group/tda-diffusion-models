
from msep.msep_cpp import MultiSpeciesExclusionProcess

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """
    dimension 3 case: projects onto a plane.
    """
    dimension = 3
    density = [1/3, 1/3, 1/3]
    length = 30

    rates_matrix_3d = np.array(
        [
            [0.0, 2.0, 2.0],
            [1.0, 0.0, 1.5],
            [1.0, 1.5, 0.0],
        ],
        dtype=np.float64,
    )

    asym_diffusion_2d = MultiSpeciesExclusionProcess(dimension=dimension, density=density, rates_matrix=rates_matrix_3d, length=length, seed=2504)

    asym_diffusion_2d.simulate(steps=100000)
    path_2d = asym_diffusion_2d.get_path_projection()

    chain = asym_diffusion_2d.get_chain()
    chain_vectors = np.cumsum(np.eye(3)[::-1][chain], axis=0)
    
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(chain_vectors[:, 0], chain_vectors[:, 1], chain_vectors[:, 2], "-o", markersize=2)
    ax1.set_xlabel("species 0")
    ax1.set_ylabel("species 1")
    ax1.set_zlabel("species 2")
    ax1.set_title("3d polymer chain")

    center = chain_vectors.mean(axis=0)
    c = center.sum()  

    x = np.linspace(chain_vectors[:, 0].min(), chain_vectors[:, 0].max(), 10)
    y = np.linspace(chain_vectors[:, 1].min(), chain_vectors[:, 1].max(), 10)
    X, Y = np.meshgrid(x, y)
    Z = c - X - Y

    ax1.plot_surface(X, Y, Z, alpha=0.25, color="orange", edgecolor="none")
    ax1.view_init(elev=25, azim=-60)
    ax1.set_box_aspect((1, 1, 1))

    ax2 = fig.add_subplot(122)
    ax2.plot(path_2d[:, 0], path_2d[:, 1], "-o", markersize=2)
    ax2.set_aspect("equal") 
    ax2.set_xlabel(r"$h_1$")
    ax2.set_ylabel(r"$h_2$")
    ax2.set_title("Projected Directed Polymer Path, d = 3")

    plt.tight_layout()
    plt.savefig("figures/projected_directed_polymer_3d.png", dpi=300)
    plt.show()

    """
    dimension 4 case: projects onto a cube.
    """
    dimension = 4
    density = [1/4, 1/4, 1/4, 1/4]
    length = 400

    rates_matrix_4d = np.array(
        [
            [0.0, 1.6, 2.3, 1.8],
            [2.4, 0.0, 2.7, 2.2],
            [1.7, 1.3, 0.0, 1.5],
            [2.2, 1.8, 2.5, 0.0],
        ],
        dtype=np.float64,
    )

    asym_diffusion_3d = MultiSpeciesExclusionProcess(dimension=dimension, density=density, rates_matrix=rates_matrix_4d, length=length, seed=2504)

    asym_diffusion_3d.simulate(steps=100000)
    path_3d = asym_diffusion_3d.get_path_projection()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(path_3d[:, 0], path_3d[:, 1], path_3d[:, 2], "-o", markersize=2)
    #print(f"x: {path_3d[-1, 0]:.3f}, y: {path_3d[-1, 1]:.3f}, z: {path_3d[-1, 2]:.3f}")
    ax.set_xlabel(r"$h_1$")
    ax.set_ylabel(r"$h_2$")
    ax.set_zlabel(r"$h_3$")
    ax.set_title("projected directed polymer path, d = 4")
    ax.set_box_aspect([1, 1, 1])

    plt.savefig("figures/projected_directed_polymer_4d.png", dpi=300)
    plt.show()
