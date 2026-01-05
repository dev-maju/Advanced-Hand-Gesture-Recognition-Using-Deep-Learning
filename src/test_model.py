import torch
from model import GestureLSTM

model = GestureLSTM()

dummy_input = torch.randn(8, 30, 63)  # batch of 8
output = model(dummy_input)

print("Output shape:", output.shape)
