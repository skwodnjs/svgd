import numpy as np
import torch
import time
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy.stats import gaussian_kde

from draw import plot_2d_contour
from model.svgd import SVGD

# min f(x) subj to g(x) >= 0

def f(particles):
    return 1 / 2 * (particles[:, 0] ** 2 + particles[:, 1] ** 2)

def g(particles):
    return particles[:, 1]

def log_p(particles):
    if type(particles) == torch.Tensor:
        return - f(particles) - 1 * torch.log(particles[:, 1])
    else:
        return - f(particles) - 1 * np.log(particles[:, 1])

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

def main(x0, sampler, max_iter, save=False):
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
    p1 = plot_2d_contour(ax, f, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100)
    p2, = ax.plot(x0[:,0], x0[:,1], '.', alpha=0.8, markersize=5, color='C2', zorder=10)
    p3 = plot_2d_contour(ax, g, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100, zero=True)
    artists.append([p1, p2, p3])

    ## method
    x = x0
    total_time = 0

    for i in range(max_iter):
        start = time.time()
        x = sampler.step(x)
        step_time = time.time() - start

        p1 = plot_2d_contour(ax, f, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100)
        p2, = ax.plot(x.detach()[:,0], x.detach()[:,1], '.', alpha=0.8, markersize=5, color='C2', zorder=10)
        p3 = plot_2d_contour(ax, g, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100, zero=True)
        artists.append([p1, p2, p3])

        plot_time = time.time() - step_time - start

        print(f"iter: {i+1:04d} \t step time: {step_time:.7f}  \t plot time: {plot_time:.7f}")
        print()
        total_time += step_time

    ## print output
    print(f'runtime: {total_time:.7f} sec')
    start = time.time()
    best = best_particle(x)
    best_particle_time = time.time() - start
    print(f'best particles: ({best[0]:.7f}, {best[1]:.7f}) \t time: {best_particle_time:.7f}')
    print(f'f(x): { - log_p(best[None, :])[0]}')

    result = f'runtime: {total_time:.4f} sec  /  best particles: ( {best[0]:.4f}, {best[1]:.4f} )  /  f(x): { - f(best[None, :])[0]:.7f}'
    plt.figtext(0.5, 0.01, result, ha="center", fontsize=10)

    ani = ArtistAnimation(fig, artists, interval= 10, repeat=True)
    if save:
        ani.save('animation/animation.gif', writer='pillow')
    plt.show()

if __name__ == "__main__":
    dim = 2
    NUM_PARTICLES = 50

    x0 = torch.zeros(NUM_PARTICLES, dim, requires_grad=False)
    x0[:,0] = torch.randn(NUM_PARTICLES, requires_grad=False)
    x0[:,1] = torch.randn(NUM_PARTICLES, requires_grad=False) + 3
    x0[:, 1] = torch.where(x0[:, 1] < 0, - x0[:, 1] + 1, x0[:, 1])

    sampler = SVGD(log_p, stepsize=0.01, alpha = 100)
    max_iter = 1000

    save = True
    main(x0, sampler, max_iter, save=save)