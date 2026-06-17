"""
neural_nets.py
==============
PyTorch-based neural network models for QSAR regression.

Models
------
1. QSAR_MLP      — Fully connected network with BatchNorm + Dropout
2. QSAR_CNN1D    — 1D Convolutional network over fingerprint bit vectors

Both implement a fit/predict interface compatible with scikit-learn pipelines.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# PyTorch Network Definitions
# ────────────────────────────────────────────────────────────────────────────

def build_mlp_network(
    input_dim: int,
    hidden_dims: List[int] = [1024, 512, 256, 128],
    dropout: float = 0.3,
    output_dim: int = 1,
) -> "torch.nn.Module":
    """
    Build a deep MLP with BatchNorm and Dropout.

    Architecture:
    Input → [Linear → BatchNorm → ReLU → Dropout] × n_layers → Linear

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list of int
        Sizes of hidden layers. Default: [1024, 512, 256, 128].
    dropout : float
        Dropout probability. Default 0.3.
    output_dim : int
        Output size. 1 for regression.

    Returns
    -------
    torch.nn.Sequential
    """
    import torch.nn as nn

    layers = []
    in_dim = input_dim
    for h in hidden_dims:
        layers += [
            nn.Linear(in_dim, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def build_cnn1d_network(
    input_dim: int,
    n_filters: int = 128,
    kernel_size: int = 7,
    n_conv_layers: int = 3,
    dropout: float = 0.3,
) -> "torch.nn.Module":
    """
    Build a 1D CNN for molecular fingerprint inputs.

    Treats the fingerprint as a 1D sequence of bits, applies convolutional
    filters to learn local bit patterns, then pools and regresses.

    Parameters
    ----------
    input_dim : int
        Fingerprint length (e.g. 2048 for standard Morgan FP).
    n_filters : int
        Number of convolutional filters.
    kernel_size : int
        Filter width.
    n_conv_layers : int
        Number of convolutional blocks.
    dropout : float
        Dropout probability.

    Returns
    -------
    torch.nn.Module
    """
    import torch
    import torch.nn as nn

    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_ch = 1
            for i in range(n_conv_layers):
                out_ch = n_filters * (2 ** i)
                layers += [
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
                in_ch = out_ch
            self.conv = nn.Sequential(*layers)
            # Compute flattened size after convolutions + pooling
            test_x = torch.zeros(1, 1, input_dim)
            with torch.no_grad():
                flat_size = self.conv(test_x).view(1, -1).shape[1]
            self.fc = nn.Sequential(
                nn.Linear(flat_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )

        def forward(self, x):
            # x: (batch, n_bits) → (batch, 1, n_bits) for Conv1d
            x = x.unsqueeze(1)
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x).squeeze(-1)

    return CNN1D()


# ────────────────────────────────────────────────────────────────────────────
# sklearn-compatible PyTorch wrapper
# ────────────────────────────────────────────────────────────────────────────

class PyTorchQSARRegressor:
    """
    sklearn-compatible wrapper for PyTorch QSAR neural networks.

    Supports:
    - fit(X, y)
    - predict(X)
    - Built-in early stopping via validation split

    Parameters
    ----------
    model_type : str
        "mlp" or "cnn1d".
    hidden_dims : list of int
        Hidden layer sizes (MLP only).
    dropout : float
        Dropout probability.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size.
    max_epochs : int
        Maximum number of epochs.
    patience : int
        Early stopping patience (epochs without val loss improvement).
    val_fraction : float
        Fraction of training data used for validation/early stopping.
    device : str
        "cuda" to use GPU if available, "cpu" to force CPU.
    """

    def __init__(
        self,
        model_type: str = "mlp",
        hidden_dims: List[int] = [1024, 512, 256, 128],
        dropout: float = 0.3,
        lr: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 200,
        patience: int = 20,
        val_fraction: float = 0.1,
        device: str = "cuda",
        random_state: int = 42,
    ):
        self.model_type    = model_type
        self.hidden_dims   = hidden_dims
        self.dropout       = dropout
        self.lr            = lr
        self.batch_size    = batch_size
        self.max_epochs    = max_epochs
        self.patience      = patience
        self.val_fraction  = val_fraction
        self.device_str    = device
        self.random_state  = random_state
        self.model_        = None
        self.history_      = {"train_loss": [], "val_loss": []}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PyTorchQSARRegressor":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, random_split
        from sklearn.model_selection import train_test_split

        torch.manual_seed(self.random_state)
        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.device_str == "cuda"
            else "cpu"
        )
        logger.info(f"Training PyTorch {self.model_type.upper()} on {device}")

        input_dim = X.shape[1]

        if self.model_type == "mlp":
            net = build_mlp_network(input_dim, self.hidden_dims, self.dropout)
        elif self.model_type == "cnn1d":
            net = build_cnn1d_network(input_dim, dropout=self.dropout)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        net = net.to(device)

        # Split train/val
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=self.val_fraction, random_state=self.random_state
        )

        def to_tensor(arr):
            return torch.tensor(arr, dtype=torch.float32)

        tr_dataset  = TensorDataset(to_tensor(X_tr), to_tensor(y_tr))
        val_dataset = TensorDataset(to_tensor(X_val), to_tensor(y_val))
        tr_loader  = DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, verbose=False
        )
        criterion = nn.MSELoss()

        best_val = float("inf")
        no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            # Train
            net.train()
            tr_loss = 0.0
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = net(Xb).squeeze(-1)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item() * len(Xb)
            tr_loss /= len(tr_dataset)

            # Validate
            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    pred = net(Xb).squeeze(-1)
                    val_loss += criterion(pred, yb).item() * len(Xb)
            val_loss /= len(val_dataset)

            scheduler.step(val_loss)
            self.history_["train_loss"].append(tr_loss)
            self.history_["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                import copy
                best_state = copy.deepcopy(net.state_dict())
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 20 == 0:
                logger.info(
                    f"Epoch {epoch:3d} | Train MSE={tr_loss:.4f} | Val MSE={val_loss:.4f}"
                )

        net.load_state_dict(best_state)
        self.model_ = net
        self.device_ = device
        logger.info(f"Training complete. Best val MSE: {best_val:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict()")
        self.model_.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device_)
        with torch.no_grad():
            preds = self.model_(X_t).squeeze(-1).cpu().numpy()
        return preds
