import os
import sys
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.model import ASLClassifier
from training import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ASLClassifier(
    input_size=config.INPUT_SIZE,
    hidden_size=config.HIDDEN_SIZE,
    num_layers=config.NUM_LAYERS,
    num_classes=config.NUM_CLASSES
).to(device)

checkpoint = "models/checkpoints/asl_lstm_epoch20.pt"
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

# Load one sample
sample_path = "data/landmarks/all/01986.npy"
x = np.load(sample_path)

x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(x)
    prediction = torch.argmax(output, dim=1)

print("Predicted class index:", prediction.item())