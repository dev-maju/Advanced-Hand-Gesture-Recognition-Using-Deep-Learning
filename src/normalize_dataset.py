import numpy as np
import os
from config import GESTURE_LABELS, SEQUENCE_LENGTH

# Absolute project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def normalize_sequence(sequence):
    """
    sequence: (30, 63)
    returns: (30, 63) normalized
    """
    normalized_seq = []

    for frame in sequence:
        frame = frame.reshape(21, 3)

        # Wrist landmark (index 0)
        wrist = frame[0]

        # Translation invariance
        frame = frame - wrist

        # Scale invariance
        scale = np.max(np.linalg.norm(frame, axis=1))
        if scale > 0:
            frame = frame / scale

        normalized_seq.append(frame.flatten())

    return np.array(normalized_seq)


for label, gesture in GESTURE_LABELS.items():
    raw_path = os.path.join(RAW_DIR, gesture)
    processed_path = os.path.join(PROCESSED_DIR, gesture)
    os.makedirs(processed_path, exist_ok=True)

    for file in os.listdir(raw_path):
        seq = np.load(os.path.join(raw_path, file))
        norm_seq = normalize_sequence(seq)

        np.save(os.path.join(processed_path, file), norm_seq)

print("Dataset normalization completed successfully.")
