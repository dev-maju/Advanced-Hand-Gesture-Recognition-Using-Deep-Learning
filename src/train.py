import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os

from dataset import GestureDataset
from model import GestureLSTM
from config import NUM_GESTURES

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001

# Dataset
dataset = GestureDataset(DATA_DIR)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# Model
device = torch.device("cpu")
model = GestureLSTM().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    val_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

# Save model
torch.save(model.state_dict(), os.path.join(BASE_DIR, "results", "gesture_lstm.pth"))
print("Model training complete. Saved to results/gesture_lstm.pth")
