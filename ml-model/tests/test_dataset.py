import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.sequence_loader import ASLDataset

dataset = ASLDataset("data/landmarks")

print("Dataset size:", len(dataset))

x, y = dataset[0]

print("Sample shape:", x.shape)   # Expected: torch.Size([30, 126])
print("Label:       ", y)

assert x.shape == (30, 126), f"Expected (30, 126), got {x.shape}"
assert x.dtype.is_floating_point, "Expected float tensor"
print("Dataset shape test passed.")