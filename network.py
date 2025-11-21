import torch
import torch.nn as nn

class SensorPlacementPolicy(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, n_actions)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight) #xavier initialization
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = torch.relu(self.fc1(x)) #non-linear transformation
        x = torch.relu(self.fc2(x)) 
        x = torch.relu(self.fc3(x))
        x = self.fc4(x) #final layer - raw scores(logits)
        return torch.softmax(x, dim=-1) #convert to probabilities that sum to 1
