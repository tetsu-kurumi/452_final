import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        x_mean = x.mean(dim=1)  # simplify: mean pooling over time
        return self.fc(x_mean).squeeze()