
from msep.msep_cpp import MultiSpeciesExclusionProcess

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

if __name__ == "__main__":

    """
    simple demonstration of three species polymer chain
    """
    dimension = 3
    density = [1/3, 1/3, 1/3]
    length = 300

    length = 30
    max_simulation_steps = 100

    """
    diagonal is zero & off diagonal entries are positive so particles move in a certain direction,
    creating an asymmetry in the diffusion process
    """
    rates_matrix_3d = np.array(
        [
            [0.0, 2.0, 2.0],
            [1.0, 0.0, 1.5],
            [1.0, 1.5, 0.0],
        ],
        dtype=np.float64,
    )

    ad_chain1 = MultiSpeciesExclusionProcess(dimension=dimension, density=density, rates_matrix=rates_matrix_3d, length=length, seed=2504, shuffle=False)
    history = ad_chain1.simulate(steps=max_simulation_steps, store_history=True, get_projection=False)
    
    ad_chain2 = MultiSpeciesExclusionProcess(dimension=dimension, density=density, rates_matrix=rates_matrix_3d, length=length, seed=2504, shuffle=False)
    proj_history = ad_chain2.simulate(steps=max_simulation_steps, store_history=True, get_projection=True)

    fig_1 = plt.figure(figsize=(12, 5))
    plt.subplots_adjust(bottom=0.2)

    ax1_1 = fig_1.add_subplot(121, projection="3d")
    ax2_1 = fig_1.add_subplot(122)

    n_frames = len(history)

    chain_history = np.empty((n_frames, history.shape[1] + 1, 3), dtype=float)

    for i in range(n_frames):
        steps_3d = np.eye(3)[history[i]]

        chain_history[i, 0] = 0.0
        chain_history[i, 1:] = np.cumsum(steps_3d, axis=0)

    x_min, x_max = chain_history[:, :, 0].min(), chain_history[:, :, 0].max()
    y_min, y_max = chain_history[:, :, 1].min(), chain_history[:, :, 1].max()
    z_min, z_max = chain_history[:, :, 2].min(), chain_history[:, :, 2].max()

    h1_min, h1_max = proj_history[:, :, 0].min(), proj_history[:, :, 0].max()
    h2_min, h2_max = proj_history[:, :, 1].min(), proj_history[:, :, 1].max()

    chain0 = chain_history[0]

    line3d, = ax1_1.plot(chain0[:, 0], chain0[:, 1], chain0[:, 2], "-o", markersize=2)
    line2d, = ax2_1.plot(proj_history[0, :, 0], proj_history[0, :, 1], "-o", markersize=2)

    ax1_1.set_xlabel("species 0")
    ax1_1.set_ylabel("species 1")
    ax1_1.set_zlabel("species 2")
    ax1_1.set_title("3d polymer chain")

    ax2_1.set_aspect("equal")
    ax2_1.set_xlabel(r"$h_1$")
    ax2_1.set_ylabel(r"$h_2$")
    ax2_1.set_title("projected polymer path d = 3")

    ax1_1.set_xlim(x_min, x_max)
    ax1_1.set_ylim(y_min, y_max)
    ax1_1.set_zlim(z_min, z_max)

    ax2_1.set_xlim(h1_min, h1_max)
    ax2_1.set_ylim(h2_min, h2_max)

    ax1_1.set_box_aspect((x_max - x_min, y_max - y_min, z_max - z_min))
    ax1_1.plot([0, 10], [0, 10], [0, 10], "k-o")

    def update(val):
        i = int(time_slider.val)

        chain = chain_history[i]

        line3d.set_data(chain[:, 0], chain[:, 1])
        line3d.set_3d_properties(chain[:, 2])

        line2d.set_data(proj_history[i, :, 0], proj_history[i, :, 1])
        fig_1.canvas.draw_idle()

    slider_ax = plt.axes([0.25, 0.05, 0.5, 0.03])

    time_slider = Slider(ax=slider_ax, label="simulation time", valmin=0, valmax=n_frames - 1, valinit=0, valfmt="%d", valstep=1)

    time_slider.on_changed(update)

    update(0)
    plt.show()

    """
    dimension three case: projects onto a plane.
    """
    length = 300

    asym_diffusion_2d = MultiSpeciesExclusionProcess(dimension=dimension, density=density, rates_matrix=rates_matrix_3d, length=length, seed=2504)

    asym_diffusion_2d.simulate(steps=100000)
    path_2d = asym_diffusion_2d.get_path_projection()

    chain = asym_diffusion_2d.get_chain()
    chain_vectors = np.cumsum(np.eye(3)[::-1][chain], axis=0)
    
    fig_2 = plt.figure(figsize=(12, 5))

    ax1_2 = fig_2.add_subplot(121, projection="3d")
    ax1_2.plot(chain_vectors[:, 0], chain_vectors[:, 1], chain_vectors[:, 2], "-o", markersize=2)
    ax1_2.set_xlabel("species 0")
    ax1_2.set_ylabel("species 1")
    ax1_2.set_zlabel("species 2")
    ax1_2.set_title("3d polymer chain")

    center = chain_vectors.mean(axis=0)
    c = center.sum()  

    x = np.linspace(chain_vectors[:, 0].min(), chain_vectors[:, 0].max(), 10)
    y = np.linspace(chain_vectors[:, 1].min(), chain_vectors[:, 1].max(), 10)
    X, Y = np.meshgrid(x, y)
    Z = c - X - Y

    ax1_2.plot_surface(X, Y, Z, alpha=0.25, color="orange", edgecolor="none")
    ax1_2.view_init(elev=25, azim=-60)
    ax1_2.set_box_aspect((1, 1, 1))

    ax2 = fig_2.add_subplot(122)
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