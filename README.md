# Advanced-Hand-Gesture-Recognition-Using-Deep-Learning
This project implements a real-time hand gesture recognition system using MediaPipe landmarks and an LSTM-based deep learning model. Normalized temporal sequences enable robust gesture classification with 97% accuracy. The system runs in real time on a standard laptop for efficient human–computer interaction.

## Overview

This project implements a real-time dynamic hand gesture recognition system using deep learning. Hand landmarks are extracted from live webcam video using MediaPipe, and temporal gesture sequences are modeled using a two-layer LSTM network implemented in PyTorch. The system supports real-time inference on a standard laptop and achieves high classification accuracy across multiple dynamic gestures.

## Features

Real-time hand landmark detection using MediaPipe

Temporal modeling of gestures using LSTM

Translation and scale invariant landmark normalization

High accuracy (≈97%) gesture classification

Runs entirely on CPU (no external hardware required)

## Gestures Supported

- Swipe Left

- Swipe Right

- Swipe Up

- Swipe Down

- Grab

## Project Structure
project_root/
├── src/                  # All source code
│   ├── config.py
│   ├── hand_landmarks.py
│   ├── record_dataset.py
│   ├── normalize_dataset.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── realtime_inference.py
│
├── results/              # Trained model and evaluation results
│   ├── gesture_lstm.pth
│   ├── confusion_matrix.npy
│   └── classification_report.txt
│
├── webcam_test.py        # Webcam verification script
└── README.md

# Dataset Directory (IMPORTANT)

You must manually create the following folder structure before recording data:

data/
├── raw/
│   ├── swipe_left/
│   ├── swipe_right/
│   ├── swipe_up/
│   ├── swipe_down/
│   └── grab/
└── processed/


data/raw/ → stores recorded gesture sequences

data/processed/ → stores normalized gesture sequences

These folders are not included in the repository and must be created locally.

## Installation

Install required dependencies:

pip install opencv-python mediapipe numpy matplotlib scikit-learn torch torchvision torchaudio

## Webcam Test

Verify webcam access before starting:

### python webcam_test.py


A live video window should open.

## Training the Model

Record gesture data:

### python src/record_dataset.py


Normalize the dataset:

### python src/normalize_dataset.py


Train the LSTM model:

### python src/train.py


The trained model will be saved in the results/ folder.

## Evaluation

Evaluate model performance:

### python src/evaluate.py


This generates:

Confusion matrix

Precision, recall, and F1-score report

## Real-Time Inference

Run live gesture recognition:

### python src/realtime_inference.py


Perform gestures in front of the webcam to see predictions in real time.

## Results

Accuracy: ~97%

Model: Two-layer LSTM

Inference: Real-time on CPU

## Future Improvements

Gesture confidence smoothing

Multi-hand support

Attention-based temporal modeling

Mobile deployment (TFLite)

## License

This project is intended for academic and educational use.
