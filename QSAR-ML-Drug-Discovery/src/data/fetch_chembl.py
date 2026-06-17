"""
fetch_chembl.py
===============
Fetch bioactivity data from the ChEMBL database using the
chembl_webresource_client REST API.

Usage
-----
    from src.data.fetch_chembl import fetch_bioactivity_data

    df = fetch_bioactivity_data(
        target_chembl_id="CHEMBL220",   # Acetylcholinesterase
        bioactivity_type="IC50",
        output_path="data/raw/ache_raw.csv"
    )
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_bioactivity_data(
    target_chembl_id: str = "CHEMBL220",
    bioactivity_type: str = "IC50",
    output_path: Optional[str] = None,
    max_results: int = 10_000,
) -> pd.DataFrame:
    """
    Fetch bioactivity data for a given ChEMBL target.

    Parameters
    ----------
    target_chembl_id : str
        ChEMBL ID of the protein target (e.g. "CHEMBL220" for AChE).
        Find targets at: https://www.ebi.ac.uk/chembl/
    bioactivity_type : str
        Type of bioactivity measurement: "IC50", "Ki", "EC50", "Kd".
    output_path : str, optional
        Path to save the raw CSV file. If None, data is not saved.
    max_results : int
        Maximum number of records to fetch (default 10,000).

    Returns
    -------
    pd.DataFrame
        Raw bioactivity data with columns:
        molecule_chembl_id, canonical_smiles, standard_type,
        standard_value, standard_units, assay_chembl_id, target_name.

    Examples
    --------
    >>> df = fetch_bioactivity_data("CHEMBL220", "IC50")
    >>> print(df.shape)
    (8234, 7)
    """
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise ImportError(
            "Install chembl_webresource_client:\n"
            "  pip install chembl-webresource-client"
        )

    logger.info(
        f"Fetching {bioactivity_type} data for target {target_chembl_id}..."
    )

    activity = new_client.activity
    target = new_client.target

    # ── 1. Confirm target exists ──────────────────────────────────────────
    target_info = target.get(target_chembl_id)
    target_name = target_info.get("pref_name", target_chembl_id)
    logger.info(f"Target: {target_name}")

    # ── 2. Fetch bioactivity records ──────────────────────────────────────
    res = activity.filter(
        target_chembl_id=target_chembl_id,
        standard_type=bioactivity_type,
    ).only(
        [
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_type",
            "standard_value",
            "standard_units",
            "standard_relation",
            "assay_chembl_id",
            "assay_description",
        ]
    )

    # Convert QuerySet to list (triggers network call)
    logger.info("Downloading records (this may take 1–2 minutes)...")
    start = time.time()
    records = list(res[:max_results])
    elapsed = time.time() - start
    logger.info(f"Downloaded {len(records)} records in {elapsed:.1f}s")

    # ── 3. Build DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame.from_records(records)
    df["target_chembl_id"] = target_chembl_id
    df["target_name"] = target_name

    # ── 4. Optionally save ────────────────────────────────────────────────
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved raw data to {output_path}")

    return df


def list_common_targets() -> pd.DataFrame:
    """
    Return a DataFrame of commonly used QSAR benchmark targets.

    Returns
    -------
    pd.DataFrame
        Table with columns: ChEMBL_ID, Name, Disease_Area, Num_Compounds_approx
    """
    targets = [
        ("CHEMBL220",  "Acetylcholinesterase",          "Alzheimer's disease",  8000),
        ("CHEMBL279",  "MAP kinase ERK2",                "Cancer",               3500),
        ("CHEMBL205",  "Carbonic anhydrase II",          "Glaucoma / Epilepsy",  6000),
        ("CHEMBL204",  "Thrombin",                       "Cardiovascular",       5000),
        ("CHEMBL240",  "COX-2",                          "Inflammation",         3000),
        ("CHEMBL217",  "Adenosine A2a receptor",         "Parkinson's disease",  4000),
        ("CHEMBL325",  "Dopamine D3 receptor",           "Schizophrenia",        2500),
        ("CHEMBL1862", "EGFR (epidermal growth factor)", "Cancer",               7000),
        ("CHEMBL2147", "BRAF kinase",                    "Melanoma",             2000),
        ("CHEMBL4282", "JAK1",                           "Autoimmune",           1500),
    ]
    return pd.DataFrame(
        targets,
        columns=["ChEMBL_ID", "Target_Name", "Disease_Area", "Approx_Compounds"],
    )
