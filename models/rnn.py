import torch
import torch.nn as nn

class UncertaintyRNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2):
        super(UncertaintyRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        _, h_n = self.rnn(x)
        out = h_n[-1]
        return self.fc(out).squeeze()
