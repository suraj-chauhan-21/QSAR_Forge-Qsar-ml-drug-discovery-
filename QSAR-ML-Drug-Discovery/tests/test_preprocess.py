"""
tests/test_preprocess.py
========================
Unit tests for src.data.preprocess module.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import ic50_to_pic50, preprocess_bioactivity, activity_class


# ── ic50_to_pic50 ────────────────────────────────────────────────────────────

class TestIC50ToPIC50:
    def test_1nm_gives_9(self):
        assert abs(ic50_to_pic50(1.0) - 9.0) < 1e-6

    def test_1um_gives_6(self):
        # 1 µM = 1000 nM → pIC50 = 6
        assert abs(ic50_to_pic50(1000.0) - 6.0) < 1e-6

    def test_1mm_gives_3(self):
        assert abs(ic50_to_pic50(1e6) - 3.0) < 1e-6

    def test_monotonically_decreasing(self):
        # Higher IC50 → lower pIC50
        vals = [10, 100, 1000, 10000]
        pics = [ic50_to_pic50(v) for v in vals]
        assert pics == sorted(pics, reverse=True)

    def test_output_type_is_float(self):
        result = ic50_to_pic50(100.0)
        assert isinstance(result, float)


# ── preprocess_bioactivity ───────────────────────────────────────────────────

@pytest.fixture
def sample_raw_df():
    """Minimal synthetic ChEMBL-like dataframe."""
    return pd.DataFrame({
        "canonical_smiles": [
            "CCO",           # valid ethanol
            "c1ccccc1",      # valid benzene
            "",              # empty SMILES → should be removed
            "invalid_smi",   # will fail → depends on RDKit but preprocess doesn't validate
            "CCO",           # duplicate of row 0
            "CCCO",          # valid propanol
        ],
        "standard_value": [100.0, 1000.0, 50.0, 200.0, 80.0, np.nan],
        "standard_units": ["nM", "nM", "nM", "nM", "nM", "nM"],
        "standard_relation": ["=", "=", "=", "=", "=", "="],
    })


class TestPreprocessBioactivity:
    def test_removes_empty_smiles(self, sample_raw_df):
        result = preprocess_bioactivity(sample_raw_df, verbose=False)
        assert "" not in result["canonical_smiles"].values

    def test_removes_nan_values(self, sample_raw_df):
        result = preprocess_bioactivity(sample_raw_df, verbose=False)
        assert result["standard_value_nM"].isna().sum() == 0

    def test_deduplicates_smiles(self, sample_raw_df):
        result = preprocess_bioactivity(sample_raw_df, verbose=False)
        assert result["canonical_smiles"].duplicated().sum() == 0

    def test_pic50_column_exists(self, sample_raw_df):
        result = preprocess_bioactivity(sample_raw_df, verbose=False)
        assert "pIC50" in result.columns

    def test_pic50_range_filtering(self, sample_raw_df):
        result = preprocess_bioactivity(
            sample_raw_df, pic50_min=5.0, pic50_max=12.0, verbose=False
        )
        assert (result["pIC50"] >= 5.0).all()
        assert (result["pIC50"] <= 12.0).all()

    def test_pic50_is_correct(self):
        # 100 nM → pIC50 = -log10(100e-9) = 7.0
        df = pd.DataFrame({
            "canonical_smiles": ["CCO"],
            "standard_value": [100.0],
            "standard_units": ["nM"],
            "standard_relation": ["="],
        })
        result = preprocess_bioactivity(df, verbose=False)
        assert abs(result["pIC50"].iloc[0] - 7.0) < 1e-5

    def test_removes_ambiguous_relations(self):
        df = pd.DataFrame({
            "canonical_smiles": ["CCO", "CCN"],
            "standard_value": [100.0, 200.0],
            "standard_units": ["nM", "nM"],
            "standard_relation": ["=", ">"],
        })
        result = preprocess_bioactivity(df, verbose=False)
        assert len(result) == 1  # Only "=" kept


# ── activity_class ───────────────────────────────────────────────────────────

class TestActivityClass:
    def test_active_label(self):
        df = pd.DataFrame({"pIC50": [7.0, 8.0]})
        result = activity_class(df, active_threshold=6.0)
        assert (result["activity_class"] == "active").all()

    def test_inactive_label(self):
        df = pd.DataFrame({"pIC50": [4.0, 4.5]})
        result = activity_class(df, inactive_threshold=5.0)
        assert (result["activity_class"] == "inactive").all()

    def test_intermediate_label(self):
        df = pd.DataFrame({"pIC50": [5.5]})
        result = activity_class(df, active_threshold=6.0, inactive_threshold=5.0)
        assert result["activity_class"].iloc[0] == "intermediate"
