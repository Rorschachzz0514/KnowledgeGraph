import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Mapper(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Mapper, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)