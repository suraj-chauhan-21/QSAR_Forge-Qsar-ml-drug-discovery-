"""
tests/test_features.py
======================
Unit tests for src.features.fingerprints and src.features.descriptors.
"""

import numpy as np
import pytest

VALID_SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]  # ethanol, benzene, aspirin
INVALID_SMILES = ["not_a_molecule", "ZZZZ"]
MIXED_SMILES = VALID_SMILES + INVALID_SMILES


# ── Fingerprints ─────────────────────────────────────────────────────────────

class TestMorganFingerprints:
    def test_import(self):
        from src.features.fingerprints import compute_morgan_fingerprints
        assert callable(compute_morgan_fingerprints)

    def test_output_shape(self):
        from src.features.fingerprints import compute_morgan_fingerprints
        X = compute_morgan_fingerprints(VALID_SMILES, radius=2, n_bits=2048)
        assert X.shape == (3, 2048)

    def test_custom_n_bits(self):
        from src.features.fingerprints import compute_morgan_fingerprints
        X = compute_morgan_fingerprints(VALID_SMILES, radius=2, n_bits=1024)
        assert X.shape == (3, 1024)

    def test_binary_values(self):
        from src.features.fingerprints import compute_morgan_fingerprints
        X = compute_morgan_fingerprints(VALID_SMILES)
        assert set(np.unique(X)).issubset({0, 1})

    def test_invalid_smiles_zero_row(self):
        from src.features.fingerprints import compute_morgan_fingerprints
        X = compute_morgan_fingerprints(INVALID_SMILES, n_bits=2048)
        # Invalid SMILES should produce zero rows
        for i in range(len(INVALID_SMILES)):
            assert X[i].sum() == 0

    def test_different_smiles_different_fingerprints(self):
        from src.features.fingerprints import compute_morgan_fingerprints
        X = compute_morgan_fingerprints(["CCO", "c1ccccc1"])
        assert not np.array_equal(X[0], X[1])

    def test_valid_mask_returned(self):
        from src.features.fingerprints import compute_morgan_fingerprints
        X, mask = compute_morgan_fingerprints(
            MIXED_SMILES, return_invalid_mask=True
        )
        assert mask.sum() == len(VALID_SMILES)
        assert (~mask).sum() == len(INVALID_SMILES)


class TestMACCSKeys:
    def test_output_shape(self):
        from src.features.fingerprints import compute_maccs_keys
        X = compute_maccs_keys(VALID_SMILES)
        assert X.shape == (3, 167)

    def test_binary_values(self):
        from src.features.fingerprints import compute_maccs_keys
        X = compute_maccs_keys(VALID_SMILES)
        assert set(np.unique(X)).issubset({0, 1})


class TestCombinedFingerprints:
    def test_output_dimension(self):
        from src.features.fingerprints import compute_combined_fingerprints
        X, names = compute_combined_fingerprints(
            VALID_SMILES, morgan_n_bits=2048, include_maccs=True
        )
        expected_cols = 2048 + 167
        assert X.shape == (3, expected_cols)
        assert len(names) == expected_cols


# ── Descriptors ──────────────────────────────────────────────────────────────

class TestRDKitDescriptors:
    def test_output_shape(self):
        from src.features.descriptors import compute_rdkit_descriptors
        X, names = compute_rdkit_descriptors(VALID_SMILES)
        assert X.shape[0] == 3
        assert len(names) == X.shape[1]
        assert X.shape[1] > 100  # should be ~200 descriptors

    def test_no_nan_in_output(self):
        from src.features.descriptors import compute_rdkit_descriptors
        X, _ = compute_rdkit_descriptors(VALID_SMILES, handle_nan="median")
        assert not np.isnan(X).any()

    def test_lipinski_descriptors(self):
        from src.features.descriptors import compute_lipinski_descriptors, LIPINSKI_DESCRIPTORS
        X, names = compute_lipinski_descriptors(VALID_SMILES)
        assert X.shape == (3, len(LIPINSKI_DESCRIPTORS))
        assert "MolWt" in names
        assert "MolLogP" in names

    def test_mw_reasonable(self):
        from src.features.descriptors import compute_lipinski_descriptors
        X, names = compute_lipinski_descriptors(["CCO"])  # ethanol, MW=46.07
        mw_idx = names.index("MolWt")
        assert 45.0 < X[0, mw_idx] < 48.0

    def test_benzene_aromatic_ring(self):
        from src.features.descriptors import compute_rdkit_descriptors
        X, names = compute_rdkit_descriptors(["c1ccccc1"],
                                              descriptor_names=["NumAromaticRings"])
        ar_idx = names.index("NumAromaticRings")
        assert X[0, ar_idx] == 1.0


class TestLipinskiRO5:
    def test_drug_like_aspirin(self):
        from src.features.descriptors import (
            compute_lipinski_descriptors, descriptors_to_dataframe,
            lipinski_rule_of_five
        )
        smiles = ["CC(=O)Oc1ccccc1C(=O)O"]  # aspirin
        X, names = compute_lipinski_descriptors(smiles)
        df = descriptors_to_dataframe(X, names, smiles)
        df_ro5 = lipinski_rule_of_five(df)
        assert df_ro5["drug_like"].iloc[0] == 1  # aspirin is drug-like
