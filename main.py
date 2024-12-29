import numpy as np
import torch
import time
import os
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy.stats import gaussian_kde

from draw import plot_2d_contour
from svgd import SVGD

# Problem statement:
# \min_{x \in \mathbb{R}^2} f(x) = (\sqrt{\frac{x_1^2}{3} + \frac{x_2^2}{2}} - 3)^2

def log_p(particles):
    return - ((particles[:, 0] ** 2 / 3 + particles[:, 1] ** 2 / 2) ** 1/2 - 3) ** 2

def best_particle(particles):
    """
    particles: numpy array of shape (num_of_particles, dim)
        - num_of_particles: particle의 개수
        - dim: 각 particle의 차원

    Returns:
    - best_particle: 확률 밀도가 가장 높은 particle
    """
    particles = particles.detach().numpy()
    kde = gaussian_kde(particles.T)
    densities = kde(particles.T)

    best_idx = np.argmax(densities)
    best_particle = particles[best_idx]

    return best_particle

def main_save(x0, sampler, max_iter):
    ## save
    dir = "frames"

    bound_x = 10
    bound_y = 10

    ## plot setting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis([-bound_x, bound_x, -bound_y, bound_y])
    ax.set_aspect('equal')
    ax.grid(True)

    ## init state plot
    plot_2d_contour(ax, log_p, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100)
    ax.plot(x0.detach()[:,0], x0.detach()[:,1], '.', alpha=0.8, markersize=5, color='C2', zorder=10)

    plt.savefig(f"{dir}/frame0000.png", dpi=300)
    ax.clear()

    ## method
    x = x0
    total_time = 0

    for i in range(max_iter):
        start = time.time()
        x = sampler.step(x)
        step_time = time.time() - start

        ## plot setting
        ax.axis([-bound_x, bound_x, -bound_y, bound_y])
        ax.set_aspect('equal')
        ax.grid(True)

        ## plot
        plot_2d_contour(ax, log_p, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100)
        ax.plot(x.detach()[:,0], x.detach()[:,1], '.', alpha=0.8, markersize=5, color='C2', zorder=10)

        plt.savefig(f"{dir}/frame{i+1:04d}.png", dpi=300)
        ax.clear()

        plot_time = time.time() - step_time - start

        print(f"iter: {i+1:04d} \t step time: {step_time:.7f}  \t plot time: {plot_time:.7f}")
        print()
        total_time += step_time

    ## print output
    print(f'runtime: {total_time:.7f}')
    start = time.time()
    best = best_particle(x)
    best_particle_time = time.time() - start
    print(f'best particles: ({best[0]:.7f}, {best[1]:.7f}) \t time: {best_particle_time:.7f}')
    print(f'analytical solution: ({2 + np.sqrt(3.5):.7f}, {2.5:.7f}), '
          f'({2 - np.sqrt(3.5):.7f}, {2.5:.7f})')

    ## image rendering
    start = time.time()
    image_files = sorted([os.path.join(dir, file) for file in os.listdir(dir) if file.endswith(".png")])
    frames = [Image.open(image) for image in image_files]
    frames[0].save(
        "animation.gif",
        save_all=True,
        append_images=frames,
        duration=100,  # ms
        loop=0,
    )
    save_time = time.time() - start
    print(f'save time: {save_time:.7f}')

def main_show(x0, sampler, max_iter):
    ## show
    bound_x = 10
    bound_y = 10

    ## plot setting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis([-bound_x, bound_x, -bound_y, bound_y])
    ax.set_aspect('equal')
    ax.grid(True)
    artists = []

    ## init state plot
    p1 = plot_2d_contour(ax, log_p, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100)
    p2, = ax.plot(x0[:,0], x0[:,1], '.', alpha=0.8, markersize=5, color='C2', zorder=10)
    artists.append([p1, p2])

    ## method
    x = x0
    total_time = 0

    for i in range(max_iter):
        start = time.time()
        x = sampler.step(x)
        step_time = time.time() - start

        p1 = plot_2d_contour(ax, log_p, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100)
        p2, = ax.plot(x.detach()[:,0], x.detach()[:,1], '.', alpha=0.8, markersize=5, color='C2', zorder=10)
        artists.append([p1, p2])

        plot_time = time.time() - step_time - start

        print(f"iter: {i+1:04d} \t step time: {step_time:.7f}  \t plot time: {plot_time:.7f}")
        print()
        total_time += step_time

    ## print output
    print(f'runtime: {total_time:.7f}')
    start = time.time()
    best = best_particle(x)
    best_particle_time = time.time() - start
    print(f'best particles: ({best[0]:.7f}, {best[1]:.7f}) \t time: {best_particle_time:.7f}')
    print(f'f(x): { - log_p(best[None, :])[0]}')

    ani = ArtistAnimation(fig, artists, interval= 10, repeat=True)
    plt.show()

if __name__ == "__main__":
    # seed = 1000
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    dim = 2
    NUM_PARTICLES = 200

    x0 = torch.zeros(NUM_PARTICLES, dim, requires_grad=False)
    x0[:,0] = torch.randn(NUM_PARTICLES, requires_grad=False) * 5
    x0[:,1] = torch.randn(NUM_PARTICLES, requires_grad=False) * 5

    sampler = SVGD(log_p, stepsize=0.5, alpha = 100)
    max_iter = 500

    save = False
    if save:
        main_save(x0, sampler, max_iter)
    else:
        main_show(x0, sampler, max_iter)