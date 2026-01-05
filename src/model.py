import torch
import torch.nn as nn
from config import FEATURE_VECTOR_SIZE, NUM_GESTURES

class GestureLSTM(nn.Module):
    def __init__(self):
        super(GestureLSTM, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=FEATURE_VECTOR_SIZE,
            hidden_size=128,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(64, NUM_GESTURES)

    def forward(self, x):
        # x: (batch, 30, 63)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Use last time step
        x = x[:, -1, :]

        x = self.fc(x)
        return x
