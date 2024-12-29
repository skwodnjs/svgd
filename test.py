import torch
import numpy as np

a = torch.tensor([[2., 1.],
                  [2., 1.]])
b = torch.tensor([[[1, 2], [2, 3]],
                   [[2, 3], [3, 4]]])
print(a.unsqueeze(2) * b)