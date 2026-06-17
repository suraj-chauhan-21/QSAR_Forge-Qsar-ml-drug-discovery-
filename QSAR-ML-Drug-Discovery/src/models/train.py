"""
train.py
========
Train and benchmark 20+ ML models on QSAR data with 5-fold cross-validation.

Usage
-----
    from src.models.train import run_benchmark

    results_df = run_benchmark(X, y, cv_folds=5, random_state=42)
    print(results_df.sort_values("R2_test", ascending=False).to_string())

CLI
---
    python -m src.models.train --features data/processed/morgan_fps.csv \
                               --target pIC50 \
                               --output results/reports/benchmark.csv
"""

import argparse
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Model Registry — All 20+ models
# ────────────────────────────────────────────────────────────────────────────

def get_all_models() -> Dict[str, object]:
    """
    Build and return a dictionary of all QSAR models.

    Returns
    -------
    dict
        Keys are model names; values are (unfitted) sklearn-compatible estimators.
    """
    from sklearn.linear_model import (
        LinearRegression, Ridge, Lasso, ElasticNet,
        BayesianRidge, HuberRegressor,
    )
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import (
        RandomForestRegressor, ExtraTreesRegressor,
        GradientBoostingRegressor, AdaBoostRegressor,
        BaggingRegressor, VotingRegressor, StackingRegressor,
        HistGradientBoostingRegressor,
    )
    from sklearn.neural_network import MLPRegressor

    try:
        import xgboost as xgb
        xgb_model = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, verbosity=0, random_state=42,
        )
    except ImportError:
        xgb_model = None
        logger.warning("XGBoost not installed; skipping.")

    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500, num_leaves=63, learning_rate=0.05,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            n_jobs=-1, verbose=-1, random_state=42,
        )
    except ImportError:
        lgb_model = None
        logger.warning("LightGBM not installed; skipping.")

    try:
        from catboost import CatBoostRegressor
        cat_model = CatBoostRegressor(
            iterations=500, depth=8, learning_rate=0.05,
            random_seed=42, verbose=0,
        )
    except ImportError:
        cat_model = None
        logger.warning("CatBoost not installed; skipping.")

    rf = RandomForestRegressor(n_estimators=300, max_features="sqrt",
                               n_jobs=-1, random_state=42)

    models = {
        # ── Linear models ────────────────────────────────────────────────
        "LinearRegression":   LinearRegression(),
        "Ridge":              Ridge(alpha=1.0),
        "Lasso":              Lasso(alpha=0.01, max_iter=5000),
        "ElasticNet":         ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "BayesianRidge":      BayesianRidge(),
        "HuberRegressor":     HuberRegressor(max_iter=300),

        # ── Kernel / instance-based ───────────────────────────────────────
        "SVR_RBF":            SVR(kernel="rbf", C=10.0, epsilon=0.1),
        "SVR_Linear":         SVR(kernel="linear", C=1.0),
        "KNN":                KNeighborsRegressor(n_neighbors=7, metric="euclidean", n_jobs=-1),

        # ── Tree models ───────────────────────────────────────────────────
        "DecisionTree":       DecisionTreeRegressor(max_depth=10, random_state=42),

        # ── Ensemble (bagging) ────────────────────────────────────────────
        "RandomForest":       rf,
        "ExtraTrees":         ExtraTreesRegressor(n_estimators=300, max_features="sqrt",
                                                   n_jobs=-1, random_state=42),
        "Bagging":            BaggingRegressor(n_estimators=50, random_state=42, n_jobs=-1),

        # ── Boosting ──────────────────────────────────────────────────────
        "GradientBoosting":   GradientBoostingRegressor(n_estimators=300,
                                                          learning_rate=0.05,
                                                          max_depth=5, random_state=42),
        "AdaBoost":           AdaBoostRegressor(n_estimators=200,
                                                 learning_rate=0.1, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(
                                   max_iter=300, learning_rate=0.05,
                                   max_leaf_nodes=31, random_state=42),

        # ── Neural Networks ───────────────────────────────────────────────
        "MLP":                MLPRegressor(hidden_layer_sizes=(512, 256, 128),
                                            activation="relu", max_iter=300,
                                            random_state=42, early_stopping=True,
                                            validation_fraction=0.1),

        # ── Meta ensembles ────────────────────────────────────────────────
        "VotingEnsemble":     VotingRegressor(estimators=[
                                  ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
                                  ("gb", GradientBoostingRegressor(n_estimators=100, random_state=42)),
                              ]),
    }

    # Add GPU-accelerated models if available
    if xgb_model is not None:
        models["XGBoost"] = xgb_model
    if lgb_model is not None:
        models["LightGBM"] = lgb_model
    if cat_model is not None:
        models["CatBoost"] = cat_model

    # Stacking (uses previously built rf + fast GBM as base learners)
    from sklearn.linear_model import Ridge as RidgeMeta
    stacking_estimators = [
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("gb", GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]
    if xgb_model is not None:
        stacking_estimators.append(("xgb", xgb.XGBRegressor(
            n_estimators=100, verbosity=0, random_state=42)))
    models["StackingEnsemble"] = StackingRegressor(
        estimators=stacking_estimators,
        final_estimator=RidgeMeta(),
        cv=3,
        n_jobs=-1,
    )

    return models


# ────────────────────────────────────────────────────────────────────────────
# Metrics helper
# ────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R², RMSE, and MAE."""
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    return {"R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4)}


# ────────────────────────────────────────────────────────────────────────────
# Main benchmark function
# ────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
    scale_features: bool = True,
    model_subset: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Train all 20+ models and return a benchmark comparison table.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_compounds, n_features).
    y : np.ndarray
        Target vector (n_compounds,) — pIC50 values.
    test_size : float
        Fraction of data held out for final test evaluation.
    cv_folds : int
        Number of cross-validation folds on the training set.
    random_state : int
        Random seed for reproducibility.
    scale_features : bool
        Whether to apply StandardScaler (important for SVR, KNN, MLP).
    model_subset : list of str, optional
        If provided, only train models with these names.
    output_path : str, optional
        If provided, save benchmark CSV to this path.

    Returns
    -------
    pd.DataFrame
        Benchmark table sorted by R²_test (descending).
        Columns: Model, R2_test, RMSE_test, MAE_test,
                 CV_R2_mean, CV_R2_std, Train_time_s
    """
    # ── Split ─────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        f"Train size: {len(X_train)} | Test size: {len(X_test)} | "
        f"Features: {X.shape[1]}"
    )

    # ── Optional scaling ──────────────────────────────────────────────────
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    # ── Get models ────────────────────────────────────────────────────────
    all_models = get_all_models()
    if model_subset is not None:
        all_models = {k: v for k, v in all_models.items() if k in model_subset}

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = []

    # ── Train and evaluate each model ─────────────────────────────────────
    for name, model in all_models.items():
        logger.info(f"Training {name}...")
        try:
            t0 = time.time()

            # Cross-validation on training set
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=kf, scoring="r2", n_jobs=-1,
            )

            # Final fit on full training set
            model.fit(X_train, y_train)
            elapsed = time.time() - t0

            # Evaluate on held-out test set
            y_pred = model.predict(X_test)
            test_metrics = compute_metrics(y_test, y_pred)

            results.append({
                "Model": name,
                "R2_test":    test_metrics["R2"],
                "RMSE_test":  test_metrics["RMSE"],
                "MAE_test":   test_metrics["MAE"],
                "CV_R2_mean": round(cv_scores.mean(), 4),
                "CV_R2_std":  round(cv_scores.std(), 4),
                "Train_time_s": round(elapsed, 2),
            })
            logger.info(
                f"  {name}: R²={test_metrics['R2']:.3f}, "
                f"RMSE={test_metrics['RMSE']:.3f}, "
                f"CV R²={cv_scores.mean():.3f}±{cv_scores.std():.3f} "
                f"({elapsed:.1f}s)"
            )

        except Exception as e:
            logger.error(f"  {name} FAILED: {e}")
            results.append({
                "Model": name, "R2_test": np.nan, "RMSE_test": np.nan,
                "MAE_test": np.nan, "CV_R2_mean": np.nan, "CV_R2_std": np.nan,
                "Train_time_s": np.nan,
            })

    # ── Build results DataFrame ───────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("R2_test", ascending=False)
    results_df = results_df.reset_index(drop=True)
    results_df.index += 1  # 1-based ranking

    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK RESULTS (sorted by R² test)")
    logger.info("=" * 70)
    logger.info(results_df.to_string())

    if output_path is not None:
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path)
        logger.info(f"\nSaved benchmark results to {output_path}")

    return results_df


# ────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Run QSAR ML benchmark with 20+ models"
    )
    parser.add_argument(
        "--features", type=str, required=True,
        help="Path to feature CSV (rows=compounds, cols=features + 'pIC50')"
    )
    parser.add_argument(
        "--target", type=str, default="pIC50",
        help="Name of the target column (default: pIC50)"
    )
    parser.add_argument(
        "--output", type=str, default="results/reports/benchmark.csv",
        help="Path to save benchmark results CSV"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Fraction of data for test set (default: 0.2)"
    )
    parser.add_argument(
        "--cv_folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    y = df[args.target].values
    X = df.drop(columns=[args.target, "canonical_smiles"], errors="ignore").values

    run_benchmark(
        X, y,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
