# Model Card: XGBoost QSAR Regressor (Acetylcholinesterase pIC50)

This model card documents the top-performing model from the standard benchmark
pipeline, following the spirit of [Mitchell et al. (2019), "Model Cards for
Model Reporting"](https://arxiv.org/abs/1810.03993), adapted for a QSAR context.

## Model Details

| Field | Value |
|---|---|
| **Model type** | Gradient-boosted decision trees (XGBoost) |
| **Task** | Regression — predict pIC50 from molecular structure |
| **Target protein** | Acetylcholinesterase (ChEMBL ID: CHEMBL220) |
| **Input representation** | Morgan fingerprint (radius=2, 2048 bits) |
| **Framework** | xgboost==2.0.x |
| **Training script** | `src/models/train.py::run_benchmark` |
| **License** | MIT (same as repository) |

## Intended Use

This model is intended for **educational and exploratory research purposes** —
demonstrating the QSAR modeling workflow described in the accompanying
tutorial. It is **not validated for clinical, regulatory, or production drug
discovery decisions**.

Appropriate uses:
- Teaching QSAR / cheminformatics ML methodology
- Prioritizing compounds for *further* experimental validation in an academic
  screening context
- Benchmarking new featurization or modeling methods against a known baseline

Inappropriate uses:
- Sole basis for clinical or regulatory decisions
- Predicting activity for chemotypes far outside the AChE training distribution
  (e.g., biologics, peptides, inorganic compounds)
- Safety/toxicity claims (this model predicts potency only, not safety)

## Training Data

- **Source**: ChEMBL database, IC50 bioactivity records for CHEMBL220
- **Size**: ~3,000–8,000 unique compounds after preprocessing (varies by
  ChEMBL release version)
- **Preprocessing**: see `src/data/preprocess.py` — deduplication, unit
  standardization, IC50 → pIC50 conversion, outlier filtering (pIC50 ∈ [3, 12])
- **Train/test split**: 80/20 random split, seed=42

## Hyperparameters

```python
XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
```

For the Optuna-tuned variant, see the hyperparameter search space in
`notebooks/QSAR_ML_Models.ipynb`, Section 9.

## Evaluation

Evaluated on a held-out 20% test split (never used during training or
cross-validation):

| Metric | Value (example run) |
|---|---|
| R² (test) | ~0.73 |
| RMSE (test) | ~0.72 pIC50 units |
| MAE (test) | ~0.55 pIC50 units |
| 5-fold CV R² (train) | ~0.72 ± 0.03 |

*Exact values vary by ChEMBL data version and random seed — reproduce via the
notebook's benchmark cell for current numbers.*

## Limitations & Caveats

- **Random train/test split**: structurally similar analogs of training
  compounds may leak into the test set, inflating apparent performance
  relative to a scaffold-split evaluation.
- **Fingerprint collisions**: Morgan fingerprints hash substructures into a
  fixed 2048-bit vector; distinct substructures can collide into the same bit,
  introducing noise.
- **Domain of applicability**: predictions for compounds very dissimilar to
  the training set (low Tanimoto similarity to any training compound) should
  be treated with low confidence.
- **No explicit handling of stereochemistry** beyond what's encoded in the
  input SMILES/Morgan fingerprint settings (`useChirality=False` by default).
- **Single-target model**: this model predicts AChE potency specifically; it
  has no general "drug-likeness" or multi-target selectivity signal.

## Ethical Considerations

This model is trained on publicly available, anonymized chemical bioactivity
data (no human subjects data). As with any predictive model used in drug
discovery, predictions should be treated as hypotheses to prioritize further
experimental work, not as ground truth.

## How to Reproduce

```python
from src.data.fetch_chembl import fetch_bioactivity_data
from src.data.preprocess import preprocess_bioactivity
from src.features.fingerprints import compute_morgan_fingerprints
from src.models.train import get_all_models
from sklearn.model_selection import train_test_split

df_raw = fetch_bioactivity_data("CHEMBL220", "IC50")
df_clean = preprocess_bioactivity(df_raw)
X = compute_morgan_fingerprints(df_clean["canonical_smiles"].tolist())
y = df_clean["pIC50"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = get_all_models()["XGBoost"]
model.fit(X_train, y_train)
```

See `notebooks/QSAR_ML_Models.ipynb` for the full end-to-end pipeline.
