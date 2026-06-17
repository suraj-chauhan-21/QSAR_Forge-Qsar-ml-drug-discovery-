# data/raw/

This directory holds **unmodified data exactly as fetched from ChEMBL**.

Files here are produced by `src/data/fetch_chembl.py::fetch_bioactivity_data()`
and should never be edited by hand. Treat this directory as read-only,
versioned input — if the underlying ChEMBL data needs to change, re-run the
fetch script rather than editing a CSV in place.

## Expected contents after running the pipeline

```
acetylcholinesterase_raw.csv   # Raw IC50 bioactivity records for CHEMBL220
```

Columns typically include: `molecule_chembl_id`, `canonical_smiles`,
`standard_type`, `standard_value`, `standard_units`, `standard_relation`,
`assay_chembl_id`, `assay_description`, `target_chembl_id`, `target_name`.

## Note on git tracking

Large raw CSVs are excluded from version control by `.gitignore`
(`data/raw/*.csv`) to keep the repository lightweight. Re-fetch with:

```bash
make fetch-data
```

or programmatically:

```python
from src.data.fetch_chembl import fetch_bioactivity_data
fetch_bioactivity_data("CHEMBL220", "IC50", output_path="data/raw/acetylcholinesterase_raw.csv")
```
