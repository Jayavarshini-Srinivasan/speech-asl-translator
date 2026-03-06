ASL Recognition Pipeline

This module implements the ASL recognition system.

Pipeline:

video
 → MediaPipe hand landmark detection
 → landmark sequence extraction
 → dataset generation
 → LSTM training
 → real-time inference
 → FastAPI prediction endpoint

Input representation:
21 landmarks × (x,y,z) = 63 features per frame

Sequence length:
30 frames