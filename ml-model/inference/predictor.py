import os
import torch
import numpy as np
from collections import deque, Counter

from training.model import ASLClassifier
from training import config
from dataset.sequence_loader import normalize_hand, add_velocity


class SignPredictor:
    """
    Single-shot ASL sign predictor.

    Loads a trained checkpoint and provides both raw predictions and
    confidence-aware predictions. The feature preprocessing (normalization +
    velocity) applied here must match what was applied during training.
    """

    def __init__(self, checkpoint):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ASLClassifier(
            input_size=config.INPUT_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            num_classes=config.NUM_CLASSES
        ).to(self.device)

        self.model.load_state_dict(
            torch.load(checkpoint, map_location=self.device)
        )
        self.model.eval()

        # Build label list from sorted class folders (must match training order)
        self.labels = sorted(os.listdir(config.DATA_DIR))

    def _preprocess(self, sequence):
        """
        Apply the same feature engineering used during training.

        Parameters
        ----------
        sequence : np.ndarray, shape (30, 63) — raw MediaPipe landmarks

        Returns
        -------
        torch.Tensor, shape (1, 30, 126)
        """
        x = normalize_hand(sequence)    # (30, 63) — position/scale invariant
        x = add_velocity(x)             # (30, 126) — add motion deltas
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)

    def predict(self, sequence):
        """
        Predict the ASL sign label from a raw landmark sequence.

        Parameters
        ----------
        sequence : np.ndarray, shape (30, 63)

        Returns
        -------
        str — predicted label name
        """
        label, _ = self.predict_with_confidence(sequence)
        return label

    def predict_with_confidence(self, sequence):
        """
        Predict the ASL sign and return the associated softmax confidence.

        Parameters
        ----------
        sequence : np.ndarray, shape (30, 63)

        Returns
        -------
        (str, float) — (predicted label, confidence 0.0–1.0)
        """
        x = self._preprocess(sequence)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = probs.max(dim=1)

        return self.labels[pred_idx.item()], conf.item()


class SmoothingPredictor:
    """
    Demo-reliable wrapper around SignPredictor with temporal smoothing.

    Strategy:
    - Maintains a rolling window of the last `window` predictions.
    - A prediction is only output when:
        1. The softmax confidence of the raw prediction exceeds `confidence_thresh`.
        2. A single class holds more than `majority_thresh` of the window.
    - If the confidence is too low, the window is cleared to prevent stale votes.
    - Returns None when uncertain (no word should be displayed to the user).

    Parameters
    ----------
    base_predictor : SignPredictor
    window : int
        Number of consecutive frame-buffer predictions to consider.
    majority_thresh : float
        Fraction of the window that must agree (e.g. 0.6 = 60%).
    confidence_thresh : float
        Minimum softmax confidence to accept a single prediction.
        For 50 classes, random chance = 0.02; 0.35 is a conservative threshold.
    """

    def __init__(
        self,
        base_predictor,
        window=10,
        majority_thresh=0.6,
        confidence_thresh=0.35,
    ):
        self.predictor = base_predictor
        self.window = deque(maxlen=window)
        self.majority_thresh = majority_thresh
        self.confidence_thresh = confidence_thresh

    def predict(self, sequence):
        """
        Predict with temporal smoothing and confidence gating.

        Parameters
        ----------
        sequence : np.ndarray, shape (30, 63)

        Returns
        -------
        str or None
            The predicted word if a stable consensus was reached, else None.
        """
        label, conf = self.predictor.predict_with_confidence(sequence)

        # Low-confidence frame: reset window to avoid stale votes accumulating
        if conf < self.confidence_thresh:
            self.window.clear()
            return "Unknown"

        self.window.append(label)

        # Not enough history yet
        if len(self.window) < self.window.maxlen:
            return "Unknown"

        # Check for majority consensus
        most_common, count = Counter(self.window).most_common(1)[0]
        if count / len(self.window) >= self.majority_thresh:
            return most_common

        return None

    def reset(self):
        """Clear the prediction window (call between sign attempts)."""
        self.window.clear()