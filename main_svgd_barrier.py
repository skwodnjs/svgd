import numpy as np
import torch
import time
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from scipy.stats import gaussian_kde

from draw import plot_2d_contour
from model.svgd_barrier import SVGD_Barrier

def f(particles):
    return 1 / 2 * (particles[:, 0] ** 2 + particles[:, 1] ** 2)

def g_0(particles):
    return particles[:, 1]

def g_1(particles):
    return - particles[:, 0] ** 2 + particles[:, 1] + 3

def g_2(particles):
    return - 2/3 * particles[:, 0] + 1 + particles[:, 1]

def g_3(particles):
    return particles[:, 1] + 1/8 * particles[:, 0] ** 2 - 3/8 * particles[:, 0] + 1/2 + 9/32

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

def main(x0, g, sampler, max_iter, save=False):
    ## show
    bound_x = 10
    bound_y = 10

    ## plot setting
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axis([-bound_x, bound_x, -bound_y, bound_y])
    ax.set_aspect('equal')
    ax.grid(True)
    artists = []

    ## init state plot
    p1 = plot_2d_contour(ax, f, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100)
    p2, = ax.plot(x0[:,0], x0[:,1], '.', alpha=0.5, markersize=5, color='#e35f62', zorder=10)
    p3 = [plot_2d_contour(ax, g_i, xlim=[-bound_x, bound_x], ylim=[-bound_y, bound_y], gridsize=100, zero=True) for g_i in g]
    artists.append([p1, p2] + p3)

    ## method
    x = x0
    total_time = 0

    for i in range(max_iter):
        start = time.time()
        gamma = 1 - i / max_iter + 1e-6
        x = sampler.step(x, gamma)
        step_time = time.time() - start

        p1 = plot_2d_contour(ax, f, xlim = [-bound_x, bound_x], ylim = [-bound_y, bound_y], gridsize = 100)
        p2, = ax.plot(x.detach()[:,0], x.detach()[:,1], '.', alpha=0.5, markersize=5, color='#e35f62', zorder=10)
        p3 = [plot_2d_contour(ax, g_i, xlim=[-bound_x, bound_x], ylim=[-bound_y, bound_y], gridsize=100, zero=True) for g_i in g]
        artists.append([p1, p2] + p3)

        plot_time = time.time() - step_time - start

        print(f"iter: {i+1:04d} \t step time: {step_time:.7f}  \t plot time: {plot_time:.7f}")
        print()
        total_time += step_time

    ## print output
    print(f'runtime: {total_time:.7f} sec')
    best = best_particle(x)
    print(f'best particles: ({best[0]:.7f}, {best[1]:.7f})')
    print(f'f(x): { f(best[None, :])[0]:.7f}')
    constraint = torch.all(torch.stack([g_i(x) <= 0 for g_i in g]), dim=0)
    print(f'constraint: {torch.all(constraint)} / g(x): {g_1(best[None, :])[0]:.7f}')

    result = f'runtime: {total_time:.4f} sec  /  best particles: ( {best[0]:.4f}, {best[1]:.4f} )  /  f(x): { - f(best[None, :])[0]:.7f} / constraint: {torch.all(constraint)}'
    plt.figtext(0.5, 0.01, result, ha="center", fontsize=10)

    ani = ArtistAnimation(fig, artists, interval= 1, repeat=True)
    if save:
        ani.save('animation/animation.gif', writer='pillow')
    plt.show()

if __name__ == "__main__":
    dim = 2
    NUM_PARTICLES = 50

    torch.manual_seed(42)

    theta = torch.rand(NUM_PARTICLES) * 2 * torch.pi
    r = torch.sqrt(torch.rand(NUM_PARTICLES)) * 2

    x = r * torch.cos(theta) + 0
    y = r * torch.sin(theta) - 7

    x0 = torch.zeros(NUM_PARTICLES, 2)
    x0[:, 0] = x
    x0[:, 1] = y

    g = [g_1, g_2, g_3]

    sampler = SVGD_Barrier(f, g, stepsize=0.5, alpha = 100)
    max_iter = 250

    save = True
    main(x0, g, sampler, max_iter, save=save)