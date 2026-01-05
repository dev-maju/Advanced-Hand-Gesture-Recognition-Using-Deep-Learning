import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sample_path = os.path.join(
    BASE_DIR,
    "data", "raw", "swipe_left", "swipe_left_000.npy"
)

print("Loading:", sample_path)
x = np.load(sample_path)
print("Sample shape:", x.shape)
