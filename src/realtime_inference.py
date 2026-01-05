import cv2
import mediapipe as mp
import numpy as np
import torch
import os
from collections import deque

from model import GestureLSTM
from config import GESTURE_LABELS, SEQUENCE_LENGTH

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "results", "gesture_lstm.pth")

# Load model
device = torch.device("cpu")
model = GestureLSTM().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Frame buffer
sequence = deque(maxlen=SEQUENCE_LENGTH)

def normalize_frame(frame):
    frame = frame.reshape(21, 3)
    wrist = frame[0]
    frame = frame - wrist
    scale = np.max(np.linalg.norm(frame, axis=1))
    if scale > 0:
        frame = frame / scale
    return frame.flatten()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_text = "..."

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = normalize_frame(np.array(landmarks))
        sequence.append(landmarks)

        if len(sequence) == SEQUENCE_LENGTH:
            input_tensor = torch.tensor(sequence, dtype=torch.float32)
            input_tensor = input_tensor.unsqueeze(0)  # (1, 30, 63)

            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
                gesture_text = GESTURE_LABELS[pred]

    cv2.putText(
        frame,
        f"Gesture: {gesture_text}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.imshow("Real-Time Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
