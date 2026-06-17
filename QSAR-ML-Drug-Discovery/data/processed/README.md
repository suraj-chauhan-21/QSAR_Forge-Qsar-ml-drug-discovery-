# data/processed/

Cleaned, featurized data ready for model training. Generated from
`data/raw/` via `src/data/preprocess.py` and `src/features/`.

## Expected contents after running the pipeline

```
clean_dataset.csv       # Deduplicated compounds with canonical_smiles + pIC50
morgan_fps.csv          # canonical_smiles + 2048-bit Morgan fingerprint columns + pIC50
descriptors.csv         # canonical_smiles + ~200 RDKit descriptor columns + pIC50
combined_features.csv   # Morgan fingerprints + descriptors merged (optional)
```

These files are regenerated deterministically from `data/raw/`, so they are
also excluded from git tracking by default (see `.gitignore`). Regenerate with:

```python
from src.data.preprocess import preprocess_bioactivity
from src.features.fingerprints import compute_morgan_fingerprints
from src.features.descriptors import compute_rdkit_descriptors
# ... see notebooks/QSAR_ML_Models.ipynb Sections 3 & 5 for the full pipeline
```
