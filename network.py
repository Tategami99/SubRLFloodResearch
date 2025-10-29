import torch
import torch.nn as nn

class SensorPlacementPolicy(nn.Module):
    def __init__(self, input_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)
