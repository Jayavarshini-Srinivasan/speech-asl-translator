import os
import sys
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset.sequence_loader import ASLDataset
from training.model import ASLClassifier
from training import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Early stopping
EARLY_STOP_PATIENCE = 12


def build_weighted_loss(labels, num_classes, device):
    """
    Compute inverse-frequency class weights to penalize majority classes more.
    This directly counters class collapse toward frequent classes.
    """
    counts = Counter(labels)
    total = len(labels)
    weights = torch.tensor(
        [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)],
        dtype=torch.float32
    ).to(device)
    return nn.CrossEntropyLoss(weight=weights)


def train():

    print(f"Using device: {device}")
    print("Loading dataset...")

    # Load full dataset WITHOUT augmentation for label indexing
    full_dataset = ASLDataset(config.DATA_DIR, augment=False)
    all_labels = np.array(full_dataset.labels)

    print(f"Dataset size: {len(full_dataset)}")
    print(f"Num classes: {len(set(all_labels.tolist()))}")

    # --- Stratified split ---
    # Ensures each class is proportionally represented in both train and val.
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(all_labels)), all_labels))

    # Training set: augmentation ON; Val set: augmentation OFF
    train_dataset = ASLDataset(config.DATA_DIR, augment=True)
    val_dataset   = ASLDataset(config.DATA_DIR, augment=False)

    train_set = Subset(train_dataset, train_idx)
    val_set   = Subset(val_dataset,   val_idx)

    print(f"Train size: {len(train_set)} | Val size: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=config.BATCH_SIZE)

    # --- Model ---
    model = ASLClassifier(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES
    ).to(device)

    # --- Class-weighted loss (fixes class collapse) ---
    train_labels = [all_labels[i] for i in train_idx]
    criterion = build_weighted_loss(train_labels, config.NUM_CLASSES, device)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # --- LR Scheduler ---
    # Halves LR when val accuracy stops improving for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

    for epoch in range(config.EPOCHS):

        model.train()
        total_loss = 0.0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            loss.backward()

            # Gradient clipping — prevents LSTM gradient explosions on small data
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_accuracy = evaluate(model, val_loader)

        # Step LR scheduler based on val accuracy
        scheduler.step(val_accuracy)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # --- Early stopping + best checkpoint save ---
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best: {best_acc:.4f} — saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    print(f"\nTraining complete. Best val accuracy: {best_acc:.4f}")
    print(f"Best model saved to: {best_model_path}")


def evaluate(model, loader):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    train()