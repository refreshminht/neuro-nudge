"""Small LSTM classifier for left vs. right fist EEG trials.

Input:  (batch, time=480, channels=64)
Output: (batch, 2) logits (0 = left, 1 = right)
"""
import torch.nn as nn


class EEGLSTM(nn.Module):
    def __init__(self, n_channels=64, hidden_size=32, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)        # (batch, time, hidden)
        feat = out.mean(dim=1)       # average over time
        return self.fc(self.dropout(feat))
