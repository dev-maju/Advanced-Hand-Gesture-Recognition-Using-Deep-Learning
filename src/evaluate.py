import torch
import numpy as np
import os
from torch.utils.data import DataLoader

from dataset import GestureDataset
from model import GestureLSTM
from config import GESTURE_LABELS

from sklearn.metrics import confusion_matrix, classification_report

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_PATH = os.path.join(BASE_DIR, "results", "gesture_lstm.pth")

# Load dataset
dataset = GestureDataset(DATA_DIR)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

# Load model
device = torch.device("cpu")
model = GestureLSTM().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        outputs = model(x)
        _, preds = torch.max(outputs, 1)

        y_true.extend(y.numpy())
        y_pred.extend(preds.numpy())

# Metrics
labels = list(GESTURE_LABELS.keys())
target_names = [GESTURE_LABELS[i] for i in labels]

cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=target_names)

print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# Save results
np.save(os.path.join(BASE_DIR, "results", "confusion_matrix.npy"), cm)

with open(os.path.join(BASE_DIR, "results", "classification_report.txt"), "w") as f:
    f.write(report)

print("\nEvaluation results saved in results/")
