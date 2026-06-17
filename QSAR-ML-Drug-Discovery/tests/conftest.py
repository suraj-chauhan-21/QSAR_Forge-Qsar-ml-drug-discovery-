"""
conftest.py
===========
Shared pytest fixtures for the QSAR-ML test suite.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_smiles():
    """Standard test SMILES: ethanol, benzene, aspirin, caffeine."""
    return [
        "CCO",                                   # Ethanol (MW~46)
        "c1ccccc1",                              # Benzene  (MW~78)
        "CC(=O)Oc1ccccc1C(=O)O",                # Aspirin  (MW~180)
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",           # Caffeine (MW~194)
    ]


@pytest.fixture(scope="session")
def sample_morgan_fps(sample_smiles):
    """Pre-computed Morgan fingerprints for test SMILES."""
    from src.features.fingerprints import compute_morgan_fingerprints
    return compute_morgan_fingerprints(sample_smiles, radius=2, n_bits=2048)


@pytest.fixture(scope="session")
def sample_descriptors(sample_smiles):
    """Pre-computed RDKit descriptors for test SMILES."""
    from src.features.descriptors import compute_lipinski_descriptors
    X, names = compute_lipinski_descriptors(sample_smiles)
    return X, names


@pytest.fixture
def synthetic_qsar_dataset():
    """
    Synthetic QSAR dataset with 200 compounds.
    Features: 200 random binary Morgan-like bits.
    Target: pIC50 in range [4, 10] (continuous).
    """
    np.random.seed(42)
    n = 200
    n_bits = 200
    X = np.random.randint(0, 2, size=(n, n_bits)).astype(np.float32)
    y = 4.0 + 6.0 * np.random.rand(n)   # pIC50 in [4, 10]
    return X, y
