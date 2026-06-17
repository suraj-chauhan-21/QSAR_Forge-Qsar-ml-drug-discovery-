"""
preprocess.py
=============
Clean raw ChEMBL bioactivity data and convert IC50 → pIC50.

Pipeline
--------
1. Drop rows missing SMILES or IC50 values
2. Remove ambiguous measurements (standard_relation != "=")
3. Standardize units to nM
4. Deduplicate SMILES (keep median IC50 per compound)
5. Convert IC50 (nM) → pIC50 = -log10(IC50 × 1e-9)
6. Filter to valid pIC50 range [pIC50_min, pIC50_max]

Usage
-----
    from src.data.preprocess import preprocess_bioactivity

    df_clean = preprocess_bioactivity(df_raw)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Unit conversion factors to nM ────────────────────────────────────────────
UNIT_TO_NM = {
    "nM": 1.0,
    "nm": 1.0,
    "uM": 1_000.0,
    "um": 1_000.0,
    "µM": 1_000.0,
    "µm": 1_000.0,
    "mM": 1_000_000.0,
    "mm": 1_000_000.0,
    "M":  1_000_000_000.0,
    "pM": 0.001,
    "pm": 0.001,
}


def ic50_to_pic50(ic50_nm: float) -> float:
    """
    Convert IC50 in nanomolar to pIC50.

    Parameters
    ----------
    ic50_nm : float
        IC50 value in nanomolar (nM).

    Returns
    -------
    float
        pIC50 = -log10(IC50 in molar) = -log10(IC50_nM × 1e-9)

    Examples
    --------
    >>> ic50_to_pic50(1.0)      # 1 nM → pIC50 = 9.0
    9.0
    >>> ic50_to_pic50(1000.0)   # 1 µM → pIC50 = 6.0
    6.0
    >>> ic50_to_pic50(1e6)      # 1 mM → pIC50 = 3.0
    3.0
    """
    ic50_molar = ic50_nm * 1e-9
    return -np.log10(ic50_molar)


def preprocess_bioactivity(
    df: pd.DataFrame,
    smiles_col: str = "canonical_smiles",
    value_col: str = "standard_value",
    unit_col: str = "standard_units",
    relation_col: str = "standard_relation",
    pic50_min: float = 3.0,
    pic50_max: float = 12.0,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Preprocess raw ChEMBL bioactivity data into a clean QSAR-ready dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw bioactivity DataFrame from fetch_chembl.fetch_bioactivity_data().
    smiles_col : str
        Column name containing canonical SMILES strings.
    value_col : str
        Column name containing bioactivity values (IC50, Ki, etc.).
    unit_col : str
        Column name containing measurement units.
    relation_col : str
        Column name containing the measurement relation ("=", ">", "<").
    pic50_min : float
        Minimum pIC50 to retain (removes inactive / flat compounds).
    pic50_max : float
        Maximum pIC50 to retain (removes extreme outliers).
    output_path : str, optional
        If provided, saves cleaned CSV to this path.
    verbose : bool
        Whether to print step-by-step statistics.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
        canonical_smiles, standard_value_nM, pIC50
    """

    def log(msg: str) -> None:
        if verbose:
            logger.info(msg)

    df = df.copy()
    n_start = len(df)
    log(f"[1/7] Starting with {n_start} records")

    # ── Step 1: Drop missing SMILES ───────────────────────────────────────
    df = df.dropna(subset=[smiles_col])
    df = df[df[smiles_col].str.strip() != ""]
    log(f"[2/7] After dropping missing SMILES: {len(df)} records "
        f"(removed {n_start - len(df)})")

    # ── Step 2: Drop missing or non-numeric values ────────────────────────
    df = df.dropna(subset=[value_col])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    df = df[df[value_col] > 0]
    log(f"[3/7] After dropping missing/invalid IC50 values: {len(df)} records")

    # ── Step 3: Remove ambiguous relations (keep only "=") ────────────────
    if relation_col in df.columns:
        n_before = len(df)
        df = df[df[relation_col] == "="]
        log(f"[4/7] After removing '>' and '<' relations: {len(df)} records "
            f"(removed {n_before - len(df)} ambiguous entries)")

    # ── Step 4: Standardize units to nM ──────────────────────────────────
    if unit_col in df.columns:
        df = df[df[unit_col].isin(UNIT_TO_NM.keys())].copy()
        df["standard_value_nM"] = df[value_col] * df[unit_col].map(UNIT_TO_NM)
    else:
        # Assume already in nM
        df["standard_value_nM"] = df[value_col]
    log(f"[5/7] After unit standardization: {len(df)} records in nM")

    # ── Step 5: Deduplicate SMILES (keep median IC50) ─────────────────────
    n_before = len(df)
    df = (
        df.groupby(smiles_col, as_index=False)
        .agg(standard_value_nM=("standard_value_nM", "median"))
    )
    log(f"[6/7] After deduplication: {len(df)} unique compounds "
        f"(removed {n_before - len(df)} duplicates)")

    # ── Step 6: Convert IC50 → pIC50 ─────────────────────────────────────
    df["pIC50"] = df["standard_value_nM"].apply(ic50_to_pic50)

    # ── Step 7: Filter valid pIC50 range ─────────────────────────────────
    n_before = len(df)
    df = df[(df["pIC50"] >= pic50_min) & (df["pIC50"] <= pic50_max)]
    log(f"[7/7] After pIC50 range filter [{pic50_min}, {pic50_max}]: "
        f"{len(df)} compounds (removed {n_before - len(df)} outliers)")

    df = df.reset_index(drop=True)

    # ── Summary ────────────────────────────────────────────────────────────
    if verbose:
        logger.info("\n── Dataset Summary ──────────────────────")
        logger.info(f"  Total compounds : {len(df)}")
        logger.info(f"  pIC50 mean      : {df['pIC50'].mean():.2f}")
        logger.info(f"  pIC50 std       : {df['pIC50'].std():.2f}")
        logger.info(f"  pIC50 min       : {df['pIC50'].min():.2f}")
        logger.info(f"  pIC50 max       : {df['pIC50'].max():.2f}")
        logger.info(f"  pIC50 median    : {df['pIC50'].median():.2f}")

    # ── Save ────────────────────────────────────────────────────────────────
    if output_path is not None:
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned data to {output_path}")

    return df


def activity_class(
    df: pd.DataFrame,
    pic50_col: str = "pIC50",
    active_threshold: float = 6.0,
    inactive_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Assign activity class labels for binary/ternary classification tasks.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a pIC50 column.
    pic50_col : str
        Name of the pIC50 column.
    active_threshold : float
        pIC50 >= active_threshold → "active" (label=1)
    inactive_threshold : float
        pIC50 < inactive_threshold → "inactive" (label=0)
        Between thresholds → intermediate (label=2; often excluded)

    Returns
    -------
    pd.DataFrame
        DataFrame with additional 'activity_class' and 'activity_label' columns.
    """
    conditions = [
        df[pic50_col] >= active_threshold,
        df[pic50_col] < inactive_threshold,
    ]
    choices = ["active", "inactive"]
    df = df.copy()
    df["activity_class"] = np.select(conditions, choices, default="intermediate")
    df["activity_label"] = np.select(
        [df["activity_class"] == "active", df["activity_class"] == "inactive"],
        [1, 0],
        default=2,
    )
    return df
