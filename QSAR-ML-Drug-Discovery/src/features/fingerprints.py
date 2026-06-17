"""
fingerprints.py
===============
Generate molecular fingerprints from SMILES strings using RDKit.

Supported fingerprint types
---------------------------
- Morgan / ECFP  (circular fingerprints, configurable radius and bit length)
- MACCS Keys     (166 predefined structural keys)
- RDKit FP       (Daylight-style path fingerprint)
- Topological Torsion
- Atom Pairs

Usage
-----
    from src.features.fingerprints import compute_morgan_fingerprints

    X = compute_morgan_fingerprints(smiles_list, radius=2, n_bits=2048)
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors

logger = logging.getLogger(__name__)


def smiles_to_mol(smiles: str) -> Optional[object]:
    """Convert a SMILES string to an RDKit Mol object (returns None on failure)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def compute_morgan_fingerprints(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = False,
    use_features: bool = False,
    return_invalid_mask: bool = False,
) -> np.ndarray:
    """
    Compute Morgan (circular) fingerprints for a list of SMILES.

    Morgan fingerprints encode the local chemical environment of each atom
    up to `radius` bonds. At radius=2, they correspond to ECFP4 fingerprints.

    Parameters
    ----------
    smiles_list : list of str
        List of SMILES strings.
    radius : int
        Morgan radius (default=2 → ECFP4; use 3 for ECFP6).
    n_bits : int
        Fingerprint length in bits (default=2048).
    use_chirality : bool
        Whether to include chirality information.
    use_features : bool
        If True, use pharmacophoric features (→ FCFP fingerprints).
    return_invalid_mask : bool
        If True, also return a boolean mask of valid SMILES.

    Returns
    -------
    np.ndarray
        Shape (n_compounds, n_bits). Rows for invalid SMILES are zeros.

    Examples
    --------
    >>> smiles = ["CCO", "c1ccccc1", "invalid_smiles"]
    >>> fps = compute_morgan_fingerprints(smiles, radius=2, n_bits=2048)
    >>> fps.shape
    (3, 2048)
    """
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    valid_mask = np.ones(len(smiles_list), dtype=bool)
    n_invalid = 0

    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is None:
            valid_mask[i] = False
            n_invalid += 1
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits,
            useChirality=use_chirality,
            useFeatures=use_features,
        )
        fps[i] = np.array(fp)

    if n_invalid > 0:
        logger.warning(
            f"{n_invalid}/{len(smiles_list)} SMILES failed to parse → zero rows"
        )

    if return_invalid_mask:
        return fps, valid_mask
    return fps


def compute_maccs_keys(
    smiles_list: List[str],
    return_invalid_mask: bool = False,
) -> np.ndarray:
    """
    Compute 166-bit MACCS structural keys for a list of SMILES.

    MACCS keys indicate presence/absence of 166 predefined substructures.
    They are interpretable because each bit has a known chemical meaning.

    Parameters
    ----------
    smiles_list : list of str
    return_invalid_mask : bool

    Returns
    -------
    np.ndarray
        Shape (n_compounds, 167). Bit 0 is unused by convention.
    """
    n_bits = 167
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    valid_mask = np.ones(len(smiles_list), dtype=bool)
    n_invalid = 0

    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is None:
            valid_mask[i] = False
            n_invalid += 1
            continue
        fp = MACCSkeys.GenMACCSKeys(mol)
        fps[i] = np.array(fp)

    if n_invalid > 0:
        logger.warning(f"{n_invalid} invalid SMILES in MACCS computation")

    if return_invalid_mask:
        return fps, valid_mask
    return fps


def compute_rdkit_fingerprints(
    smiles_list: List[str],
    n_bits: int = 2048,
    max_path: int = 5,
) -> np.ndarray:
    """
    Compute RDKit path-based (Daylight-style) fingerprints.

    These are hashed path fingerprints that enumerate all paths in the
    molecular graph up to max_path bonds.

    Parameters
    ----------
    smiles_list : list of str
    n_bits : int
        Fingerprint length (default 2048).
    max_path : int
        Maximum path length to enumerate (default 5).

    Returns
    -------
    np.ndarray
        Shape (n_compounds, n_bits)
    """
    from rdkit.Chem import RDKFingerprint

    fps = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    n_invalid = 0

    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is None:
            n_invalid += 1
            continue
        fp = RDKFingerprint(mol, maxPath=max_path, fpSize=n_bits)
        fps[i] = np.array(fp)

    if n_invalid > 0:
        logger.warning(f"{n_invalid} invalid SMILES in RDKit FP computation")
    return fps


def compute_combined_fingerprints(
    smiles_list: List[str],
    morgan_radius: int = 2,
    morgan_n_bits: int = 2048,
    include_maccs: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute and horizontally concatenate multiple fingerprint types.

    Parameters
    ----------
    smiles_list : list of str
    morgan_radius : int
    morgan_n_bits : int
    include_maccs : bool
        Whether to append 167-bit MACCS keys to the Morgan fingerprints.

    Returns
    -------
    X : np.ndarray
        Combined feature matrix.
    feature_names : list of str
        Names for each column.
    """
    parts = []
    names = []

    # Morgan
    morgan = compute_morgan_fingerprints(
        smiles_list, radius=morgan_radius, n_bits=morgan_n_bits
    )
    parts.append(morgan)
    names += [f"Morgan_{i}" for i in range(morgan_n_bits)]

    # MACCS
    if include_maccs:
        maccs = compute_maccs_keys(smiles_list)
        parts.append(maccs)
        names += [f"MACCS_{i}" for i in range(167)]

    X = np.hstack(parts)
    return X, names


def fingerprint_to_dataframe(
    fps: np.ndarray,
    smiles_list: List[str],
    prefix: str = "FP",
) -> pd.DataFrame:
    """
    Wrap a fingerprint matrix as a labeled pandas DataFrame.

    Parameters
    ----------
    fps : np.ndarray
        Shape (n_compounds, n_bits).
    smiles_list : list of str
    prefix : str
        Column name prefix.

    Returns
    -------
    pd.DataFrame
    """
    n_bits = fps.shape[1]
    cols = [f"{prefix}_{i}" for i in range(n_bits)]
    df = pd.DataFrame(fps, columns=cols)
    df.insert(0, "canonical_smiles", smiles_list)
    return df
