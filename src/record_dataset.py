import cv2
import mediapipe as mp
import numpy as np
import os
from config import GESTURE_LABELS, SEQUENCE_LENGTH

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)

print("Gesture labels:")
for k, v in GESTURE_LABELS.items():
    print(f"{k}: {v}")

gesture_id = int(input("Enter gesture ID to record: "))
gesture_name = GESTURE_LABELS[gesture_id]
gesture_path = os.path.join(DATA_DIR, gesture_name)
os.makedirs(gesture_path, exist_ok=True)

sample_count = len(os.listdir(gesture_path))
print(f"Recording samples for gesture: {gesture_name}")

while True:
    input("Press ENTER to start recording one gesture...")
    sequence = []

    while len(sequence) < SEQUENCE_LENGTH:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            sequence.append(landmarks)

            cv2.putText(frame, f"Recording {gesture_name} ({len(sequence)}/{SEQUENCE_LENGTH})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Dataset Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    sequence = np.array(sequence)
    filename = f"{gesture_name}_{sample_count:03d}.npy"
    np.save(os.path.join(gesture_path, filename), sequence)
    sample_count += 1

    print(f"Saved: {filename}")

cap.release()
cv2.destroyAllWindows()
