"""
evaluate.py
===========
Model evaluation, plotting, and SHAP explainability for QSAR models.

Usage
-----
    from src.models.evaluate import evaluate_model, plot_actual_vs_predicted

    metrics = evaluate_model(model, X_test, y_test)
    plot_actual_vs_predicted(y_test, y_pred, model_name="XGBoost")
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Evaluate a fitted model on a test set and return key metrics.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
    X_test : np.ndarray
    y_test : np.ndarray
    model_name : str

    Returns
    -------
    dict with keys: R2, RMSE, MAE, model_name
    """
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    logger.info(
        f"{model_name}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}"
    )
    return {
        "model_name": model_name,
        "R2":   round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE":  round(mae, 4),
        "y_pred": y_pred,
    }


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot actual vs. predicted pIC50 values with R² annotation.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth pIC50 values.
    y_pred : np.ndarray
        Model predictions.
    model_name : str
        Plot title prefix.
    save_path : str, optional
        If provided, saves figure to this path.
    show : bool
        Whether to call plt.show().
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    import numpy as np

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, color="#2E86AB", edgecolors="none")

    # Diagonal line (perfect predictions)
    mn = min(y_true.min(), y_pred.min()) - 0.3
    mx = max(y_true.max(), y_pred.max()) + 0.3
    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")

    # Trendline
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(mn, mx, 200)
    ax.plot(x_line, p(x_line), "#E87722", lw=1.5, alpha=0.8, label="Trend")

    ax.set_xlabel("Actual pIC50", fontsize=13)
    ax.set_ylabel("Predicted pIC50", fontsize=13)
    ax.set_title(f"{model_name}\nR² = {r2:.3f} | RMSE = {rmse:.3f}", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved actual vs predicted plot to {save_path}")

    if show:
        plt.show()
    plt.close()


def plot_benchmark_comparison(
    results_df: pd.DataFrame,
    metric: str = "R2_test",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Horizontal bar chart comparing all models on a given metric.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of run_benchmark() — must contain 'Model' and metric columns.
    metric : str
        Column to plot. Default "R2_test".
    save_path : str, optional
    show : bool
    """
    import matplotlib.pyplot as plt

    df_sorted = results_df.dropna(subset=[metric]).sort_values(metric)
    colors = ["#2E86AB" if v >= 0.7 else "#A8DADC" if v >= 0.5 else "#E5E5E5"
              for v in df_sorted[metric]]

    fig, ax = plt.subplots(figsize=(10, 0.45 * len(df_sorted) + 1.5))
    bars = ax.barh(df_sorted["Model"], df_sorted[metric], color=colors, edgecolor="none")

    # Value labels
    for bar, val in zip(bars, df_sorted[metric]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=9)

    ax.set_xlabel(metric.replace("_", " "), fontsize=12)
    ax.set_title(f"Model Benchmark — {metric.replace('_', ' ')}", fontsize=14)
    ax.set_xlim(0, min(1.0, df_sorted[metric].max() + 0.08))
    ax.axvline(0.7, ls="--", lw=1, color="gray", alpha=0.5, label="R²=0.7 threshold")
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved benchmark comparison to {save_path}")

    if show:
        plt.show()
    plt.close()


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    model_type: str = "tree",
    n_samples: int = 500,
) -> np.ndarray:
    """
    Compute SHAP values for a trained model.

    Parameters
    ----------
    model : fitted model
        Tree-based (XGBoost, RF, LightGBM, CatBoost) or linear model.
    X : np.ndarray
        Samples to explain.
    feature_names : list of str, optional
    model_type : str
        "tree" → TreeExplainer (fast, exact for tree-based models)
        "linear" → LinearExplainer
        "kernel" → KernelExplainer (slow but model-agnostic)
    n_samples : int
        Max samples to explain (subsample if X has more rows).

    Returns
    -------
    shap_values : np.ndarray
        Shape (n_samples, n_features).
    """
    try:
        import shap
    except ImportError:
        raise ImportError("Install shap: pip install shap")

    if len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X_explain = X[idx]
    else:
        X_explain = X

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
    elif model_type == "linear":
        explainer = shap.LinearExplainer(model, X_explain)
        shap_values = explainer.shap_values(X_explain)
    else:
        explainer = shap.KernelExplainer(model.predict, X_explain[:100])
        shap_values = explainer.shap_values(X_explain)

    return shap_values


def plot_shap_summary(
    model,
    X_test: np.ndarray,
    feature_names: Optional[List[str]] = None,
    model_type: str = "tree",
    max_display: int = 20,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Generate a SHAP summary (beeswarm) plot.

    Each dot represents one compound. Color shows feature value (red=high,
    blue=low). X position shows SHAP impact on predicted pIC50.

    Parameters
    ----------
    model : fitted model
    X_test : np.ndarray
    feature_names : list of str, optional
    model_type : str
        "tree", "linear", or "kernel".
    max_display : int
        Number of top features to display. Default 20.
    save_path : str, optional
    show : bool
    """
    import shap
    import matplotlib.pyplot as plt

    shap_values = compute_shap_values(
        model, X_test, feature_names=feature_names, model_type=model_type
    )

    plt.figure(figsize=(10, 0.4 * max_display + 2))
    shap.summary_plot(
        shap_values,
        X_test[:len(shap_values)],
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Feature Importance (Top Features)", fontsize=13)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved SHAP summary to {save_path}")

    if show:
        plt.show()
    plt.close()
