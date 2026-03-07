import torch
import torch.nn as nn


class ASLClassifier(nn.Module):
    """
    Bidirectional LSTM for ASL sign classification.

    Changes from original:
    - Uses global mean+max temporal pooling instead of last-token only.
      This captures the full temporal dynamics of the gesture.
    - Added LSTM recurrent dropout (only when num_layers > 1).
    - fc layer input adjusted: hidden_size * 2 (bidir) * 2 (mean+max) = hidden_size * 4.
    """

    def __init__(self, input_size=126, hidden_size=128, num_layers=2, num_classes=50):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(0.4)

        # mean-pool + max-pool → concat → 4x hidden_size
        self.fc = nn.Linear(hidden_size * 4, num_classes)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)                  # (B, T, hidden*2)

        mean_pool = lstm_out.mean(dim=1)             # (B, hidden*2)
        max_pool = lstm_out.max(dim=1).values        # (B, hidden*2)

        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (B, hidden*4)

        pooled = self.dropout(pooled)

        output = self.fc(pooled)                    # (B, num_classes)

        return output