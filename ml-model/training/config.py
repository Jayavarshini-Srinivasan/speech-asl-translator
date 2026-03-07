# Training hyperparameters

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001


# Dataset path
DATA_DIR = "data/landmarks"


# Model save location
CHECKPOINT_DIR = "models/checkpoints"


# Model architecture parameters
# INPUT_SIZE = 126 because:
#   63 (normalized hand landmarks) + 63 (velocity deltas) = 126
INPUT_SIZE = 126
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 50