import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_window):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_window)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_lstm_out = lstm_out[:, -1, :]
        output = self.fc(last_lstm_out)
        return output
    