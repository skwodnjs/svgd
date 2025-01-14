import math

import numpy as np
import torch
import torch.nn as nn

class SVGD_Barrier(nn.Module):
    def __init__(self, f, g, stepsize=1e-1, alpha = 20., M = 1000, iter = 100):
        super(SVGD_Barrier, self).__init__()
        self.log_p = lambda particles: - f(particles)
        self.g = g
        self.stepsize = stepsize
        self.alpha = alpha
        self.M = M
        self.iter = iter

    def step(self, particle, gamma):
        grad = self.svgd_get_gradient(lambda particles: self.log_p(particles) + gamma * sum(torch.log(g(particles)) for g in self.g), particle)
        dx = grad.detach().clone() * self.stepsize
        new_data = particle.data + torch.clip(dx, -self.M * gamma, self.M * gamma)  ### negative sign for constraint SVGD
        for i in range(len(new_data)):
            if any(g(new_data[i].unsqueeze(0)) > -1e-2 / gamma for g in self.g):
                new_data[i] = particle.data[i]
            else:
                j = 0
                while any(g(new_data[i].unsqueeze(0)) > 0 for g in self.g):
                    if j > self.iter:
                        new_data[i] = particle.data[i]
                        break
                    new_data[i] = 1/2 * new_data[i] + 1/2 * particle.data[i]
                    j += 1
        particle.data = new_data  ### negative sign for constraint SVGD
        return particle

    def kernel(self, particles):
        pairwise_subj = particles[:, None] - particles
        pairwise_distance = torch.norm(pairwise_subj, dim=2)  # n x n matrix

        n = particles.size(0)  # number of parcitles
        median = np.median(pairwise_distance.detach().numpy())
        h = median / math.log(n)

        kernel_matrix = torch.exp(- pairwise_distance ** 2 / (h * 1. + 1e-6))
        kernel_matrix_grad = kernel_matrix.unsqueeze(2) * pairwise_subj

        return kernel_matrix, kernel_matrix_grad

    def svgd_get_gradient(self, log_p, particles):
        n = particles.size(0)  # number of parcitles
        particles = particles.detach().requires_grad_(True)

        log_prob = log_p(particles)  #  log_prob = [log p (particle) for parcitle in particles]
        log_prob_grad = torch.autograd.grad(log_prob.sum(), particles, allow_unused=True, retain_graph=True)[0]  # n x dim

        kernel_matrix, kernel_matrix_grad = self.kernel(particles)

        term1 = torch.einsum("ij,ik->jk", kernel_matrix, log_prob_grad)
        term2 = kernel_matrix_grad.sum(dim=1)

        svgd_gradient = (term1 + term2) / n
        return svgd_gradient