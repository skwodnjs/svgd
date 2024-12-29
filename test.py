import math

import torch
import numpy as np

def g(particles):
    return 2 * particles[:, 0] - particles[:, 1]

def f(particles):
    return particles[:, 1]

h = g + f
a = torch.tensor([[-2., 1.],
                  [-2., 1.]], requires_grad=True)
print(h(a))