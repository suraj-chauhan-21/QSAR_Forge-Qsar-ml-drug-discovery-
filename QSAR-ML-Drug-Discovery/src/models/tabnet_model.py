"""
tabnet_model.py
===============
TabNet QSAR regressor using pytorch-tabnet.

TabNet uses sequential attention to select which features to use at each
decision step, providing built-in interpretability alongside strong performance.

Reference
---------
Arik, S.O., & Pfister, T. (2021).
TabNet: Attentive Interpretable Tabular Learning. AAAI 2021.
arXiv:1908.07442

Usage
-----
    from src.models.tabnet_model import train_tabnet

    model, history = train_tabnet(X_train, y_train, X_val, y_val)
    preds = model.predict(X_test)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def train_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_d: int = 64,
    n_a: int = 64,
    n_steps: int = 5,
    gamma: float = 1.5,
    lambda_sparse: float = 1e-3,
    batch_size: int = 512,
    virtual_batch_size: int = 128,
    max_epochs: int = 200,
    patience: int = 25,
    learning_rate: float = 2e-2,
    seed: int = 42,
) -> Tuple[object, Dict]:
    """
    Train a TabNet regressor on QSAR data.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training features and targets.
    X_val, y_val : np.ndarray, optional
        Validation set for early stopping. If None, 10% of training is used.
    n_d : int
        Width of the decision step output (embedding dimension). Default 64.
    n_a : int
        Width of the attention embedding. Default 64 (usually equal to n_d).
    n_steps : int
        Number of sequential attention steps. Default 5.
    gamma : float
        Coefficient for feature reusage in attention. Default 1.5.
    lambda_sparse : float
        Sparsity regularization coefficient. Default 1e-3.
    batch_size : int
        Mini-batch size for training. Default 512.
    virtual_batch_size : int
        Ghost Batch Normalization size. Must be ≤ batch_size. Default 128.
    max_epochs : int
        Maximum training epochs. Default 200.
    patience : int
        Early stopping patience (epochs without validation improvement). Default 25.
    learning_rate : float
        Initial learning rate. Default 2e-2.
    seed : int
        Random seed.

    Returns
    -------
    model : TabNetRegressor
        Fitted model with .predict(), .explain() methods.
    history : dict
        Training history dict with keys 'train_mse', 'val_mse'.

    Examples
    --------
    >>> model, history = train_tabnet(X_train, y_train, X_val, y_val)
    >>> preds = model.predict(X_test)
    >>> print(preds.shape)
    (500,)
    """
    try:
        from pytorch_tabnet.tab_model import TabNetRegressor
    except ImportError:
        raise ImportError(
            "Install pytorch-tabnet:\n"
            "  pip install pytorch-tabnet"
        )

    # Ensure 2D targets
    y_train_2d = y_train.reshape(-1, 1).astype(np.float32)
    X_train_f  = X_train.astype(np.float32)

    eval_set = None
    if X_val is not None and y_val is not None:
        y_val_2d = y_val.reshape(-1, 1).astype(np.float32)
        X_val_f  = X_val.astype(np.float32)
        eval_set = [(X_val_f, y_val_2d)]
        eval_names = ["val"]
        eval_metric = ["rmse"]
    else:
        eval_names = None
        eval_metric = ["rmse"]

    model = TabNetRegressor(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        gamma=gamma,
        lambda_sparse=lambda_sparse,
        optimizer_params={"lr": learning_rate},
        scheduler_fn=None,
        mask_type="sparsemax",
        seed=seed,
        verbose=10,
    )

    model.fit(
        X_train=X_train_f,
        y_train=y_train_2d,
        eval_set=eval_set,
        eval_name=eval_names,
        eval_metric=eval_metric,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
    )

    history = {
        "train_mse": model.history.get("loss", []),
        "val_mse":   model.history.get("val_rmse", []) if eval_set else [],
    }

    logger.info(
        f"TabNet training complete. Best epoch: {model.best_epoch}. "
        f"Best val RMSE: {model.best_cost:.4f}"
    )

    return model, history


def get_tabnet_feature_importances(
    model: object,
    X_explain: np.ndarray,
    feature_names: Optional[list] = None,
) -> np.ndarray:
    """
    Extract feature importances from a trained TabNet model.

    TabNet provides two types of importance:
    - Global: average attention masks across all samples
    - Local: per-sample masks (explain individual predictions)

    Parameters
    ----------
    model : TabNetRegressor
        Trained TabNet model.
    X_explain : np.ndarray
        Samples to explain.
    feature_names : list, optional
        If provided, returns a labeled Series.

    Returns
    -------
    importances : np.ndarray
        Global feature importances (shape: n_features,).
    """
    explain_matrix, masks = model.explain(X_explain.astype(np.float32))
    # Global importance = mean of mask_step over all samples and steps
    importances = np.mean(explain_matrix, axis=0)

    if feature_names is not None:
        import pandas as pd
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)

    return importances
