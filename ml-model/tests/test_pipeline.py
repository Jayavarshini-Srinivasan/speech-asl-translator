import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from preprocessing.mediapipe_detector import MediaPipeHandDetector


detector = MediaPipeHandDetector()

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    landmarks = detector.detect(frame)

    if landmarks is not None:
        print("Detected landmark vector:", landmarks.shape)

    cv2.imshow("ASL Hand Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()