import math

import numpy as np
import torch
import torch.nn as nn
import time

class SVGD(nn.Module):
    def __init__(self, log_p, stepsize=1e-1, alpha = 20., M = 1000):
        super(SVGD, self).__init__()
        self.log_p = log_p
        self.stepsize = stepsize
        self.alpha = alpha
        self.M = M

    def step(self, particle):
        grad = self.svgd_get_gradient(self.log_p, particle)
        dx = grad.detach().clone() * self.stepsize
        particle.data = particle.data +  torch.clip(dx, -self.M, self.M)  ### negative sign for constraint SVGD
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
        start = time.time()
        n = particles.size(0)  # number of parcitles
        particles = particles.detach().requires_grad_(True)

        log_prob = log_p(particles)  #  log_prob = [log p (particle) for parcitle in particles]
        log_prob_grad = torch.autograd.grad(log_prob.sum(), particles, allow_unused=True, retain_graph=True)[0]  # n x dim

        kernel_matrix, kernel_matrix_grad = self.kernel(particles)

        svgd_gradient = torch.zeros_like(particles, device=particles.device)

        term1 = torch.einsum("ij,ik->jk", kernel_matrix, log_prob_grad)
        term2 = kernel_matrix_grad.sum(dim=1)

        svgd_gradient = (term1 + term2) / n
        return svgd_gradient