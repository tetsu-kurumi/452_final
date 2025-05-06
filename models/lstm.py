import torch
import torch.nn as nn

class UncertaintyLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2):
        super(UncertaintyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (num_layers, batch, hidden_dim)
        out = h_n[-1]  # Last layer hidden state
        return self.fc(out).squeeze()