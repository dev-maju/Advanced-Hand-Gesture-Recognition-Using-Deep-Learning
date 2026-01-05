# src/config.py

GESTURE_LABELS = {
    0: "swipe_left",
    1: "swipe_right",
    2: "swipe_up",
    3: "swipe_down",
    4: "grab"
}

NUM_GESTURES = len(GESTURE_LABELS)
SEQUENCE_LENGTH = 30     # frames per gesture
NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 3
FEATURE_VECTOR_SIZE = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 63
