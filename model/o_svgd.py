import math

import numpy as np
import torch
import torch.nn as nn
import time

# class O_SVGD(nn.Module):
#     def __init__(self, log_p, g, stepsize=1e-1, alpha = 20., M = 1000):
#         super(O_SVGD, self).__init__()
#         self.log_p = log_p
#         self.g = g
#         self.stepsize = stepsize
#         self.alpha = alpha
#         self.M = M
#
#     def step(self, particle):
#         grad = self.svgd_get_gradient(self.log_p, particle)
#         dx = grad.detach().clone() * self.stepsize
#         particle.data = particle.data +  torch.clip(dx, -self.M, self.M)  ### negative sign for constraint SVGD
#         return particle
#
#     def kernel(self, particles):
#         pairwise_subj = particles[:, None] - particles
#         pairwise_distance = torch.norm(pairwise_subj, dim=2)  # n x n matrix
#
#         n = particles.size(0)  # number of parcitles
#         median = np.median(pairwise_distance.detach().numpy())
#         h = median / math.log(n)
#
#         kernel_matrix = torch.exp(- pairwise_distance ** 2 / (h * 1. + 1e-6))
#         kernel_matrix_grad = kernel_matrix.unsqueeze(2) * pairwise_subj
#
#         return kernel_matrix, kernel_matrix_grad
#
#     def svgd_get_gradient(self, log_p, particles):
#         start = time.time()
#         n = particles.size(0)  # number of parcitles
#         particles = particles.detach().requires_grad_(True)
#
#         log_prob = log_p(particles)  #  log_prob = [log p (particle) for parcitle in particles]
#         log_prob_grad = torch.autograd.grad(log_prob.sum(), particles, allow_unused=True, retain_graph=True)[0]  # n x dim
#
#         kernel_matrix, kernel_matrix_grad = self.kernel(particles)
#
#         svgd_gradient = torch.zeros_like(particles, device=particles.device)
#
#         term1 = torch.einsum("ij,ik->jk", kernel_matrix, log_prob_grad)
#         term2 = kernel_matrix_grad.sum(dim=1)
#
#         svgd_gradient = (term1 + term2) / n
#         return svgd_gradient

class O_SVGD(nn.Module):
    def __init__(self, log_p, g, stepsize=1e-1, alpha = 20., M = 1000):
        super(O_SVGD,self).__init__()
        self.log_p = log_p
        self.g = g
        self.stepsize = stepsize
        self.alpha = alpha
        self.M = M

    def step(self, particle):
        grad, c = self.svgd_get_gradient(self.log_p, self.g, particle)
        dx = grad.detach().clone() * self.stepsize
        particle.data = particle.data +  torch.clip(dx, -self.M, self.M)  ### negative sign for constraint SVGD
        return particle

    def get_single_particle_gradient_with_rbf_and_c(self,idx, inputs, log_prob_grad, rbf_kernel_matrix, c_list):
        n = inputs.size(0)
        d = inputs.shape[1]
        grad = None
        for j in range(n):
            K_rbf = rbf_kernel_matrix[idx, j] * torch.eye(d, device=inputs.device)
            K = (c_list[idx][0].mm(K_rbf)).mm(c_list[j][0])

            mle_term = K.mm(log_prob_grad[j].unsqueeze(1)).squeeze()
            if grad is None:
                grad = mle_term.detach().clone()
            else:
                grad = grad + mle_term.detach().clone()

            for k1 in range(d):
                for k2 in range(d):
                    grad_k = torch.autograd.grad(K[k1, k2].sum(), inputs, allow_unused=True, retain_graph=True)[0]
                    grad[k1] = grad[k1] + grad_k[j, k2]

        grad_final = grad - self.alpha * (c_list[idx][1] / c_list[idx][2].norm().pow(2)) * c_list[idx][2].squeeze()
        return grad

    def median(self,tensor):
        """
        torch.median() acts differently from np.median(). We want to simulate numpy implementation.
        """
        tensor = tensor.detach().flatten()
        tensor_max = tensor.max()[None]
        return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.

    def kernel_rbf(self,inputs):
        n = inputs.shape[0]
        pairwise_distance = torch.norm(inputs[:, None] - inputs, dim=2).pow(2)
        h = self.median(pairwise_distance) / math.log(n)
        kernel_matrix = torch.exp(-pairwise_distance / (1.*h+1e-6))

        return kernel_matrix

    def svgd_get_gradient(self, log_p, constraint, particles):
        start = time.time()
        n = particles.size(0)  # number of parcitles
        particles = particles.detach().requires_grad_(True)

        log_prob = log_p(particles)  #  log_prob = [log p (particle) for parcitle in particles]
        log_prob_grad = torch.autograd.grad(log_prob.sum(), particles, allow_unused=True, retain_graph=True)[0]
        step_1 = time.time() - start

        constraint_value = constraint(particles)

        c_list = []
        for i in range(n):
            constraint_grad = torch.autograd.grad(constraint_value[i].sum(), particles, allow_unused=True, create_graph=True)[0]
            constraint_grad = constraint_grad[i].unsqueeze(1)

            g_norm_sqr = constraint_grad.norm().pow(2)
            D = torch.eye(constraint_grad.shape[0]) - constraint_grad@constraint_grad.t() / g_norm_sqr
            c_list.append((D, constraint_value[i], constraint_grad))

        rbf_kernel_matrix = self.kernel_rbf(particles)

        svgd_gradient = torch.zeros_like(particles, device=particles.device)

        step_2 = time.time() - start - step_1
        # too
        for i in range(n):
            svgd_gradient[i, :] = self.get_single_particle_gradient_with_rbf_and_c( \
                i, particles, log_prob_grad, rbf_kernel_matrix, c_list).detach().clone()

        gradient = svgd_gradient / n

        step_3 = time.time() - start - step_2

        print(f'step 1: {step_1:.5f} \t step 2: {step_2:.5f} \t step 3: {step_3:.5f}')

        return gradient.squeeze(), constraint_value

    def svgd_get_gradient_fast(self, log_p, constraint, particles):
        n = particles.size(0)  # number of parcitles
        particles = particles.detach().requires_grad_(True)

        log_prob = log_p(particles)  #  log_prob = [log p (particle) for parcitle in particles]
        log_prob_grad = torch.autograd.grad(log_prob.sum(), particles, allow_unused=True, retain_graph=True)[0]
        constraint_value = constraint(particles)
        constraint_grad = torch.autograd.grad(constraint_value.sum(), particles, allow_unused=True, create_graph=True)[0]

        s_perp, g_para = self.project_g(log_prob_grad,  constraint_grad)
        rbf_kernel_matrix = self.kernel_rbf(particles)

        svgd_gradient = torch.zeros_like(particles, device=particles.device)
        for i in range(n):
            svgd_gradient[i, :] = self.get_single_particle_gradient_with_rbf_and_c_fast( \
                i, particles, constraint_value, constraint_grad, rbf_kernel_matrix, s_perp).detach().clone()

        gradient = svgd_gradient / n
        return gradient.squeeze(), constraint_value

    def get_single_particle_gradient_with_rbf_and_c_fast(self,idx, inputs, constraint, constraint_grad, rbf_kernel_matrix, s_perp):
        n = inputs.size(0)
        d = inputs.shape[1]
        grad = None
        for j in range(n):
            mle_term, _ = self.project_g(s_perp[None,idx,:]*rbf_kernel_matrix[idx,j],  constraint_grad[None,j,:])

            if grad is None:
                grad = mle_term.detach().clone()
            else:
                grad = grad + mle_term.detach().clone()
            grad_k = torch.autograd.grad(rbf_kernel_matrix[idx,j], inputs, allow_unused=True, retain_graph=True)[0]

            temp, _ = self.project_g(grad_k[None,j,:],  constraint_grad[None,idx,:])
            dd_grad_k, _ = self.project_g(temp,  constraint_grad[None,j,:])
            grad += dd_grad_k
        grad -= self.alpha * constraint[idx]*constraint_grad[idx,:]/constraint_grad[idx,:].norm().pow(2)

        return grad
    def project_g(self,v, dg):
        proj = torch.sum(v*dg,dim=1)/torch.sum(dg**2,dim=1)
        g_para =proj.unsqueeze(1).repeat(1,v.size(1))*dg
        g_perp = v - g_para
        return g_perp, g_para

class GD(nn.Module):
    def __init__(self, logp, g, stepsize=1e-1):
        super(GD,self).__init__()
        self.logp = logp
        self.g = g
        self.stepsize = stepsize
        self.dim = 2

    def step(self, x):
        gx = self.g(x)
        Dgx = torch.sign(gx)*self.compute_grad(x,self.g)
        x = x - self.stepsize*Dgx
        return x

    def compute_grad(self,x, model):
        x = x.requires_grad_()
        gx = torch.autograd.grad(model(x).sum(), x)[0]
        return gx.detach()