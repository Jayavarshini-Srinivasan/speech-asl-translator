import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.predictor import SignPredictor, SmoothingPredictor

CHECKPOINT = "models/checkpoints/best_model.pt"

predictor = SignPredictor(CHECKPOINT)
smoother  = SmoothingPredictor(predictor)

base = "data/landmarks"

print(f"{'Label':<20} {'Predicted':<20} {'Confidence':>10}  Smoothed")
print("-" * 65)

for label in sorted(os.listdir(base)):

    folder = os.path.join(base, label)

    if not os.path.isdir(folder):
        continue

    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    if not files:
        continue

    path = os.path.join(folder, files[0])
    sample = np.load(path)  # (30, 63) raw landmarks

    pred, conf = predictor.predict_with_confidence(sample)

    # Feed into smoother 10 times to simulate a window of identical frames
    smoother.reset()
    smoothed = None
    for _ in range(10):
        smoothed = smoother.predict(sample)

    print(f"{label:<20} {pred:<20} {conf:>10.4f}  {smoothed}")