import math

import torch
import numpy as np

def g(particles):
    return 2 * particles[:, 0] - particles[:, 1]

a = torch.tensor([[-2., 1.],
                  [-2., 1.]], requires_grad=True)
b = torch.log(torch.clip(a, min=1))
print(b.sum())
print(torch.autograd.grad(b.sum(), a))