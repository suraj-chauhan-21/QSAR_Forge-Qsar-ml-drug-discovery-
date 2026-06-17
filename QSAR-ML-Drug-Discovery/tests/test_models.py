"""
tests/test_models.py
=====================
Unit tests for src.models.train and src.models.evaluate.

These tests use a small synthetic dataset for speed — they verify the
pipeline mechanics (shapes, metrics, ranking) rather than QSAR performance.
"""

import numpy as np
import pandas as pd
import pytest


# ── get_all_models ───────────────────────────────────────────────────────────

class TestGetAllModels:
    def test_returns_dict(self):
        from src.models.train import get_all_models
        models = get_all_models()
        assert isinstance(models, dict)

    def test_contains_core_models(self):
        from src.models.train import get_all_models
        models = get_all_models()
        expected = {
            "LinearRegression", "Ridge", "Lasso", "RandomForest",
            "ExtraTrees", "SVR_RBF", "KNN", "DecisionTree",
            "GradientBoosting", "AdaBoost", "MLP",
        }
        assert expected.issubset(set(models.keys()))

    def test_at_least_15_models(self):
        from src.models.train import get_all_models
        models = get_all_models()
        # XGBoost/LightGBM/CatBoost may be unavailable in minimal CI envs,
        # but the core sklearn registry alone should be 15+.
        assert len(models) >= 15

    def test_all_models_have_fit_predict(self):
        from src.models.train import get_all_models
        models = get_all_models()
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} missing fit()"
            assert hasattr(model, "predict"), f"{name} missing predict()"


# ── compute_metrics ───────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect_prediction(self):
        from src.models.train import compute_metrics
        y = np.array([1.0, 2.0, 3.0, 4.0])
        metrics = compute_metrics(y, y)
        assert metrics["R2"] == 1.0
        assert metrics["RMSE"] == 0.0
        assert metrics["MAE"] == 0.0

    def test_metrics_keys(self):
        from src.models.train import compute_metrics
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 2.9])
        metrics = compute_metrics(y_true, y_pred)
        assert set(metrics.keys()) == {"R2", "RMSE", "MAE"}

    def test_rmse_nonnegative(self):
        from src.models.train import compute_metrics
        y_true = np.random.rand(50)
        y_pred = np.random.rand(50)
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["RMSE"] >= 0
        assert metrics["MAE"] >= 0


# ── run_benchmark (integration, small subset for speed) ───────────────────────

class TestRunBenchmark:
    def test_returns_dataframe(self, synthetic_qsar_dataset):
        from src.models.train import run_benchmark
        X, y = synthetic_qsar_dataset
        results = run_benchmark(
            X, y,
            cv_folds=2,
            model_subset=["LinearRegression", "Ridge", "RandomForest"],
        )
        assert isinstance(results, pd.DataFrame)

    def test_contains_expected_columns(self, synthetic_qsar_dataset):
        from src.models.train import run_benchmark
        X, y = synthetic_qsar_dataset
        results = run_benchmark(
            X, y, cv_folds=2,
            model_subset=["LinearRegression", "Ridge"],
        )
        expected_cols = {
            "Model", "R2_test", "RMSE_test", "MAE_test",
            "CV_R2_mean", "CV_R2_std", "Train_time_s",
        }
        assert expected_cols.issubset(set(results.columns))

    def test_sorted_by_r2_descending(self, synthetic_qsar_dataset):
        from src.models.train import run_benchmark
        X, y = synthetic_qsar_dataset
        results = run_benchmark(
            X, y, cv_folds=2,
            model_subset=["LinearRegression", "Ridge", "RandomForest", "KNN"],
        )
        r2_values = results["R2_test"].dropna().values
        assert all(r2_values[i] >= r2_values[i + 1] for i in range(len(r2_values) - 1))

    def test_row_count_matches_model_subset(self, synthetic_qsar_dataset):
        from src.models.train import run_benchmark
        X, y = synthetic_qsar_dataset
        subset = ["LinearRegression", "Ridge", "Lasso"]
        results = run_benchmark(X, y, cv_folds=2, model_subset=subset)
        assert len(results) == len(subset)


# ── evaluate_model ───────────────────────────────────────────────────────────

class TestEvaluateModel:
    def test_evaluate_fitted_model(self, synthetic_qsar_dataset):
        from sklearn.linear_model import Ridge
        from src.models.evaluate import evaluate_model

        X, y = synthetic_qsar_dataset
        model = Ridge().fit(X, y)
        metrics = evaluate_model(model, X, y, model_name="Ridge")

        assert "R2" in metrics
        assert "RMSE" in metrics
        assert "MAE" in metrics
        assert "y_pred" in metrics
        assert len(metrics["y_pred"]) == len(y)
