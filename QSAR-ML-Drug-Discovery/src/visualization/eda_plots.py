"""
eda_plots.py
============
Exploratory data analysis plots for QSAR datasets.

Includes:
- pIC50 distribution histogram
- Feature correlation heatmap
- Lipinski descriptor pairplots
- Chemical space visualization (PCA / t-SNE on fingerprints)
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_pic50_distribution(
    pic50_values: np.ndarray,
    active_threshold: float = 6.0,
    inactive_threshold: float = 5.0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot the distribution of pIC50 values with activity threshold lines.

    Parameters
    ----------
    pic50_values : np.ndarray
    active_threshold : float
        Vertical line marking the "active" cutoff.
    inactive_threshold : float
        Vertical line marking the "inactive" cutoff.
    save_path : str, optional
    show : bool
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(pic50_values, bins=40, kde=True, color="#2E86AB", ax=ax)

    ax.axvline(active_threshold, color="#2A9D8F", ls="--", lw=2,
               label=f"Active threshold (pIC50 ≥ {active_threshold})")
    ax.axvline(inactive_threshold, color="#E76F51", ls="--", lw=2,
               label=f"Inactive threshold (pIC50 < {inactive_threshold})")

    ax.set_xlabel("pIC50", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"pIC50 Distribution (n={len(pic50_values)}, "
        f"mean={np.mean(pic50_values):.2f}, std={np.std(pic50_values):.2f})",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved pIC50 distribution plot to {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_feature_correlation_heatmap(
    df_features: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot a correlation heatmap for selected (typically Lipinski) descriptors.

    Parameters
    ----------
    df_features : pd.DataFrame
    feature_cols : list of str, optional
        Subset of columns to include. If None, all numeric columns are used.
    save_path : str, optional
    show : bool
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if feature_cols is None:
        feature_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()

    corr = df_features[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(0.6 * len(feature_cols) + 2, 0.6 * len(feature_cols) + 2))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title("Feature Correlation Matrix", fontsize=13)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved correlation heatmap to {save_path}")
    if show:
        plt.show()
    plt.close()


def plot_chemical_space(
    fingerprints: np.ndarray,
    pic50_values: np.ndarray,
    method: str = "pca",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize chemical space using dimensionality reduction (PCA or t-SNE),
    colored by pIC50 value.

    Parameters
    ----------
    fingerprints : np.ndarray
        High-dimensional fingerprint matrix (e.g., Morgan FPs).
    pic50_values : np.ndarray
        Corresponding pIC50 values for coloring.
    method : str
        "pca" (fast, linear) or "tsne" (slower, nonlinear, better separation).
    save_path : str, optional
    show : bool
    """
    import matplotlib.pyplot as plt

    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        title = "Chemical Space (PCA)"
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        title = "Chemical Space (t-SNE)"
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    embedding = reducer.fit_transform(fingerprints)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=pic50_values, cmap="viridis", s=18, alpha=0.7, edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("pIC50", fontsize=11)
    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)
    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved chemical space plot to {save_path}")
    if show:
        plt.show()
    plt.close()
