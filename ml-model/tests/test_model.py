import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.model import ASLClassifier
from training import config

model = ASLClassifier(
    input_size=config.INPUT_SIZE,    # 126 after feature engineering
    hidden_size=config.HIDDEN_SIZE,
    num_layers=config.NUM_LAYERS,
    num_classes=config.NUM_CLASSES
)

# Use a random input matching the new feature dimensions: (batch=8, T=30, features=126)
x = torch.randn(8, 30, config.INPUT_SIZE)

y = model(x)

print("Input shape: ", x.shape)
print("Output shape:", y.shape)
assert y.shape == (8, config.NUM_CLASSES), f"Unexpected output shape: {y.shape}"
print("Model shape test passed.")