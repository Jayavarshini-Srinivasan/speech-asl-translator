import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import numpy as np
from tqdm import tqdm

from preprocessing.mediapipe_detector import MediaPipeHandDetector


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw_filtered")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "landmarks")

SEQUENCE_LENGTH = 30


detector = MediaPipeHandDetector()


def extract_video_landmarks(video_path):

    cap = cv2.VideoCapture(video_path)

    sequence = []

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        landmarks = detector.detect(frame)

        if landmarks is not None:
            sequence.append(landmarks)

    cap.release()

    if len(sequence) < SEQUENCE_LENGTH:
        return None

    sequence = sequence[:SEQUENCE_LENGTH]

    return np.array(sequence)


def process_dataset():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for label in os.listdir(VIDEO_DIR):

        label_dir = os.path.join(VIDEO_DIR, label)
        output_label_dir = os.path.join(OUTPUT_DIR, label)

        os.makedirs(output_label_dir, exist_ok=True)

        for video in tqdm(os.listdir(label_dir), desc=label):

            video_path = os.path.join(label_dir, video)

            sequence = extract_video_landmarks(video_path)

            if sequence is None:
                continue

            save_path = os.path.join(
                output_label_dir,
                video.replace(".mp4", ".npy")
            )

            np.save(save_path, sequence)


if __name__ == "__main__":

    print("Extracting landmarks...")

    process_dataset()

    print("Landmark extraction complete.")