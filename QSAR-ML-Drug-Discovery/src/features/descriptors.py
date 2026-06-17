"""
descriptors.py
==============
Compute RDKit 2D physicochemical descriptors from SMILES strings.

RDKit provides ~200 descriptors covering:
  - Molecular weight (MW, ExactMW)
  - Lipophilicity (MolLogP, MolMR)
  - Polar surface area (TPSA)
  - H-bond donors/acceptors (NumHDonors, NumHAcceptors)
  - Rotatable bonds (NumRotatableBonds)
  - Ring counts (RingCount, NumAromaticRings)
  - Topological / graph invariants
  - Electronic / quantum chemistry descriptors

Usage
-----
    from src.features.descriptors import compute_rdkit_descriptors

    X_desc, feat_names = compute_rdkit_descriptors(smiles_list)
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

logger = logging.getLogger(__name__)

# ── Curated subset of "Lipinski-relevant" + ADMET descriptors ────────────────
LIPINSKI_DESCRIPTORS = [
    "MolWt",             # Molecular weight
    "MolLogP",           # Lipophilicity (logP)
    "NumHDonors",        # H-bond donors
    "NumHAcceptors",     # H-bond acceptors
    "NumRotatableBonds", # Rotatable bonds (flexibility)
    "TPSA",              # Topological polar surface area
    "RingCount",         # Total ring count
    "NumAromaticRings",  # Aromatic ring count
    "NumSaturatedRings", # Saturated ring count
    "NumAliphaticRings", # Aliphatic ring count
    "FractionCSP3",      # Fraction of carbons that are sp3
    "HeavyAtomCount",    # Heavy atom count
    "NHOHCount",         # N and O H-count
    "NOCount",           # N and O count
    "NumHeteroatoms",    # Heteroatom count
    "MolMR",             # Molar refractivity
]

# ── All descriptor names ──────────────────────────────────────────────────────
ALL_DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]


def compute_rdkit_descriptors(
    smiles_list: List[str],
    descriptor_names: Optional[List[str]] = None,
    handle_nan: str = "median",
    return_invalid_mask: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute RDKit 2D molecular descriptors for a list of SMILES strings.

    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings.
    descriptor_names : list of str, optional
        Specific descriptors to compute. If None, all ~200 are computed.
        Use LIPINSKI_DESCRIPTORS for a curated 16-feature subset.
    handle_nan : str
        Strategy for NaN values: "median" (default) or "zero".
    return_invalid_mask : bool
        If True, also return a boolean mask of valid SMILES.

    Returns
    -------
    X : np.ndarray
        Shape (n_compounds, n_descriptors). NaN rows filled by handle_nan.
    feature_names : list of str
        Descriptor names.

    Examples
    --------
    >>> X, names = compute_rdkit_descriptors(["CCO", "c1ccccc1"])
    >>> X.shape
    (2, 200)

    >>> X_lip, names_lip = compute_rdkit_descriptors(
    ...     ["CCO"], descriptor_names=LIPINSKI_DESCRIPTORS
    ... )
    >>> X_lip.shape
    (1, 16)
    """
    if descriptor_names is None:
        descriptor_names = ALL_DESCRIPTOR_NAMES

    # Build lookup: name → function
    desc_funcs = {name: func for name, func in Descriptors.descList}
    selected = [(name, desc_funcs[name]) for name in descriptor_names
                if name in desc_funcs]

    n_compounds = len(smiles_list)
    n_descs = len(selected)
    X = np.full((n_compounds, n_descs), np.nan)
    valid_mask = np.ones(n_compounds, dtype=bool)
    n_invalid = 0

    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
        except Exception:
            mol = None

        if mol is None:
            valid_mask[i] = False
            n_invalid += 1
            continue

        for j, (name, func) in enumerate(selected):
            try:
                val = func(mol)
                if val is not None and np.isfinite(val):
                    X[i, j] = val
            except Exception:
                pass  # Leave as NaN

    if n_invalid > 0:
        logger.warning(
            f"{n_invalid}/{n_compounds} SMILES failed to parse → NaN rows"
        )

    # ── Handle NaN ────────────────────────────────────────────────────────
    if handle_nan == "median":
        col_medians = np.nanmedian(X, axis=0)
        for j in range(n_descs):
            nan_rows = np.isnan(X[:, j])
            X[nan_rows, j] = col_medians[j] if np.isfinite(col_medians[j]) else 0.0
    elif handle_nan == "zero":
        X = np.nan_to_num(X, nan=0.0)

    feat_names = [name for name, _ in selected]

    if return_invalid_mask:
        return X, feat_names, valid_mask
    return X, feat_names


def compute_lipinski_descriptors(smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute the 16 Lipinski + ADMET-relevant descriptors only.

    These are interpretable and sufficient for many QSAR tasks.

    Returns
    -------
    X : np.ndarray, shape (n_compounds, 16)
    feature_names : list of str, length 16
    """
    return compute_rdkit_descriptors(smiles_list, descriptor_names=LIPINSKI_DESCRIPTORS)


def descriptors_to_dataframe(
    X: np.ndarray,
    feature_names: List[str],
    smiles_list: List[str],
) -> pd.DataFrame:
    """
    Wrap descriptor matrix as a pandas DataFrame with SMILES index.

    Parameters
    ----------
    X : np.ndarray
    feature_names : list of str
    smiles_list : list of str

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(X, columns=feature_names)
    df.insert(0, "canonical_smiles", smiles_list)
    return df


def lipinski_rule_of_five(df_descriptors: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate a descriptor DataFrame with Lipinski Rule of Five compliance.

    Lipinski RO5 filters for oral drug-likeness:
    - MW ≤ 500 Da
    - LogP ≤ 5
    - HBD ≤ 5
    - HBA ≤ 10

    Parameters
    ----------
    df_descriptors : pd.DataFrame
        Must contain columns: MolWt, MolLogP, NumHDonors, NumHAcceptors

    Returns
    -------
    pd.DataFrame with additional column 'RO5_violations' (0–4).
    """
    df = df_descriptors.copy()
    df["RO5_MW"]  = (df["MolWt"] > 500).astype(int)
    df["RO5_LogP"] = (df["MolLogP"] > 5).astype(int)
    df["RO5_HBD"] = (df["NumHDonors"] > 5).astype(int)
    df["RO5_HBA"] = (df["NumHAcceptors"] > 10).astype(int)
    df["RO5_violations"] = df[["RO5_MW", "RO5_LogP", "RO5_HBD", "RO5_HBA"]].sum(axis=1)
    df["drug_like"] = (df["RO5_violations"] <= 1).astype(int)
    return df
