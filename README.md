#  QSAR_Forge |QSAR-ML Drug Discovery|

> **Quantitative Structure–Activity Relationship Modeling with 20+ Machine Learning Models**  
> *"Machine Learning on QSAR Data | XGBoost, LightGBM, CatBoost, TabNet, Neural Nets & More!"*

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/RDKit-2023.09-orange?logo=rdkit" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4-green?logo=scikit-learn" />
  <img src="https://img.shields.io/badge/XGBoost-2.0-red" />
  <img src="https://img.shields.io/badge/LightGBM-4.3-blue" />
  <img src="https://img.shields.io/badge/CatBoost-1.2-yellow" />
  <img src="https://img.shields.io/badge/PyTorch-2.2-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
</p>

---

##  Table of Contents

- [What is QSAR?](#-what-is-qsar)
- [What You Will Learn](#-what-you-will-learn)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Molecular Representations](#-molecular-representations)
- [Models Implemented](#-models-implemented)
- [Pipeline Overview](#-pipeline-overview)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
- [Results & Benchmarks](#-results--benchmarks)
- [Explainability with SHAP](#-explainability-with-shap)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)
- [Community & Policies](#-community--policies)

---

## What is QSAR?

**Quantitative Structure–Activity Relationship (QSAR)** is a computational methodology that establishes mathematical relationships between the chemical structure of a molecule and its biological or physicochemical activity.

### The Core Idea

```
Chemical Structure (SMILES) → Molecular Features (Descriptors / Fingerprints) → ML Model → Predicted Activity (pIC50 / pKi)
```

QSAR models are foundational in **computational drug discovery** because they allow researchers to:
- **Screen millions of virtual compounds** without wet-lab synthesis
- **Predict potency** (IC50, Ki, EC50) against a biological target
- **Prioritize lead compounds** for experimental testing
- **Reduce costs and time** in early-stage drug discovery

### Key Terminology

| Term | Definition |
|------|-----------|
| **IC50** | Half-maximal inhibitory concentration — concentration of a compound needed to inhibit a biological process by 50% |
| **pIC50** | Negative log₁₀ of IC50 in molar units: `pIC50 = -log10(IC50_M)`. Higher = more potent |
| **SMILES** | Simplified Molecular Input Line Entry System — a 1D string notation for chemical structures |
| **Morgan Fingerprints** | Circular fingerprints encoding the local chemical environment around each atom (ECFP family) |
| **RDKit Descriptors** | 200+ physicochemical descriptors computed from a molecular graph (MW, LogP, TPSA, HBD, HBA, etc.) |
| **MACCS Keys** | 166 predefined structural keys indicating presence/absence of specific substructures |
| **ChEMBL** | Large-scale open-access bioactivity database maintained by EMBL-EBI |

---

##  What You Will Learn

By working through this repository and the companion notebook, you will understand:

1. **Data Collection**: How to query ChEMBL for bioactivity data for any protein target using `chembl_webresource_client`
2. **Data Preprocessing**: Cleaning IC50 values, handling missing data, converting units, removing duplicates
3. **pIC50 Conversion**: Why we model pIC50 (log-transformed) instead of raw IC50 values (skewed distribution)
4. **Molecular Featurization**:
   - Morgan (circular) fingerprints via RDKit
   - RDKit 2D physicochemical descriptors
   - MACCS keys
   - Combined feature sets
5. **Train/Test Splitting**: Scaffold-aware splits vs. random splits to avoid data leakage
6. **Model Training**: 20+ regression models from classical ML to deep learning
7. **Hyperparameter Tuning**: Optuna-based Bayesian optimization
8. **Model Evaluation**: R², RMSE, MAE, and cross-validation
9. **SHAP Explainability**: Identifying which molecular features drive predictions
10. **Model Persistence**: Saving and loading trained models for inference

---

## Repository Structure

```
QSAR-ML-Drug-Discovery/
│
├── README.md                        ← You are here
├── LICENSE
├── CHANGELOG.md                     ← Version history (Keep a Changelog format)
├── CITATION.cff                     ← Machine-readable citation (GitHub "Cite" button)
├── CODE_OF_CONDUCT.md                ← Contributor Covenant
├── CONTRIBUTING.md                  ← How to contribute
├── SECURITY.md                      ← Vulnerability reporting policy
├── Makefile                         ← Common dev commands (test, lint, format, benchmark)
├── .editorconfig                     ← Cross-IDE formatting consistency
├── .gitignore
├── requirements.txt                 ← All Python dependencies
├── environment.yml                  ← Conda environment specification
├── setup.py                         ← Package installation
├── pyproject.toml                   ← Modern Python packaging
│
├── .github/
│   ├── CODEOWNERS                   ← Auto-assigns PR reviewers
│   ├── FUNDING.yml                  ← GitHub Sponsors button
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── dependabot.yml                ← Automated dependency update PRs
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── feature_request.md
│   │   ├── question.md
│   │   └── config.yml
│   └── workflows/
│       ├── ci.yml                   ← Automated testing on push/PR
│       └── lint.yml                 ← Code quality checks
│
├── data/
│   ├── raw/                         ← Original unmodified data from ChEMBL
│   │   ├── README.md
│   │   └── acetylcholinesterase_raw.csv
│   ├── processed/                   ← Cleaned, featurized data ready for ML
│   │   ├── README.md
│   │   ├── descriptors.csv          ← RDKit 2D descriptors
│   │   ├── morgan_fps.csv           ← Morgan fingerprints (2048 bits)
│   │   └── combined_features.csv    ← Merged descriptor + fingerprint matrix
│   └── external/                    ← Any additional datasets
│       └── README.md
│
├── notebooks/
│   └── QSAR_ML_Models.ipynb         ← Main tutorial notebook (20+ models)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetch_chembl.py          ← ChEMBL API querying
│   │   └── preprocess.py            ← Cleaning, IC50→pIC50, deduplication
│   ├── features/
│   │   ├── __init__.py
│   │   ├── descriptors.py           ← RDKit descriptor calculation
│   │   ├── fingerprints.py          ← Morgan / MACCS / RDKit FP generation
│   │   └── feature_selection.py     ← Variance / correlation filtering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical_ml.py          ← sklearn models
│   │   ├── gradient_boosting.py     ← XGBoost, LightGBM, CatBoost
│   │   ├── neural_nets.py           ← MLP, PyTorch models
│   │   ├── tabnet_model.py          ← TabNet (pytorch-tabnet)
│   │   ├── train.py                 ← Training loops + cross-validation
│   │   └── evaluate.py              ← Metrics, benchmark table
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── eda_plots.py             ← Distribution, correlation plots
│   │   └── results_plots.py         ← Actual vs predicted, SHAP plots
│   └── utils/
│       ├── __init__.py
│       └── helpers.py               ← Logging, file I/O, seed setting
│
├── results/
│   ├── README.md
│   ├── figures/                     ← Saved plots (PNG / SVG)
│   ├── models/                      ← Serialized trained models (.pkl, .pt)
│   └── reports/
│       └── benchmark_results.csv    ← R², RMSE, MAE for all 20+ models
│
├── tests/
│   ├── conftest.py                  ← Shared pytest fixtures
│   ├── test_preprocess.py
│   ├── test_features.py
│   └── test_models.py
│
└── docs/
    ├── QSAR_theory.md               ← Background reading
    └── model_cards/
        └── xgboost_ache_model_card.md
```

---

##  Dataset

### Source: ChEMBL Database

This tutorial uses **Acetylcholinesterase (AChE)** as the target protein — a canonical benchmark target in QSAR literature and a key drug target for Alzheimer's disease.

- **ChEMBL Target ID**: `CHEMBL220`
- **Bioactivity type**: IC50 (nM)
- **Data size**: ~4,000–10,000 compounds (varies by filter)
- **Database link**: [https://www.ebi.ac.uk/chembl/target/inspect/CHEMBL220](https://www.ebi.ac.uk/chembl/target/inspect/CHEMBL220)

### Why Acetylcholinesterase?

AChE inhibitors (donepezil, rivastigmine, galantamine) are among the few FDA-approved drugs for Alzheimer's disease. The ChEMBL AChE dataset is large, well-curated, and widely used as a QSAR benchmark — making it ideal for learning and comparison.

### Data Preprocessing Steps

```
ChEMBL IC50 data
        │
        ▼
1. Filter for IC50 assay type only
        │
        ▼
2. Remove ">" / "<" operator rows (ambiguous measurements)
        │
        ▼
3. Keep only nM unit entries; standardize to nM
        │
        ▼
4. Remove rows with missing SMILES or IC50
        │
        ▼
5. Deduplicate SMILES (keep median IC50 per compound)
        │
        ▼
6. Convert IC50 (nM) → pIC50: pIC50 = -log10(IC50 × 1e-9)
        │
        ▼
7. Filter: 3 ≤ pIC50 ≤ 12 (remove extreme outliers)
        │
        ▼
Final dataset: ~3,000–8,000 compounds with SMILES + pIC50
```

---

##  Molecular Representations

### 1. Morgan Fingerprints (ECFP-style)

Morgan fingerprints encode the local chemical environment of each atom up to radius `r`. At radius 2 they are equivalent to ECFP4 (Extended-Connectivity Fingerprints).

```python
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.MolFromSmiles("CCOc1ccccc1")
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
# → 2048-dimensional binary vector
```

**Parameters used in this project:**
- `radius = 2` (captures 2-bond neighbourhood)
- `nBits = 2048` (standard bit vector size)

### 2. RDKit 2D Physicochemical Descriptors

```python
from rdkit.Chem import Descriptors
import pandas as pd

descriptor_names = [x[0] for x in Descriptors.descList]
values = [x[1](mol) for x in Descriptors.descList]
# → ~200 numeric descriptors (MW, LogP, TPSA, HBD, HBA, RotBonds, ...)
```

Key descriptors computed:

| Descriptor | Abbreviation | Meaning |
|------------|--------------|---------|
| Molecular Weight | MW | Sum of atomic masses |
| Lipophilicity | LogP | Partition coefficient (octanol/water) |
| Topological Polar Surface Area | TPSA | Estimate of oral bioavailability |
| H-bond Donors | HBD | Count of -OH and -NH groups |
| H-bond Acceptors | HBA | Count of O and N atoms |
| Rotatable Bonds | RotBonds | Flexibility measure |
| Aromatic Rings | ArRings | Count of aromatic rings |

### 3. MACCS Keys

166 predefined structural keys, each indicating the presence or absence of a specific chemical feature (e.g., "ring size > 6", "C-N bond", etc.).

```python
from rdkit.Chem import MACCSkeys
fp = MACCSkeys.GenMACCSKeys(mol)  # → 167-bit vector (bit 0 unused)
```

---

## Models Implemented

This repository trains and benchmarks **20+ regression models** on QSAR data:

### Classical Machine Learning (scikit-learn)

| # | Model | Key Hyperparameters |
|---|-------|-------------------|
| 1 | **Linear Regression** | Baseline |
| 2 | **Ridge Regression** | `alpha` |
| 3 | **Lasso Regression** | `alpha` |
| 4 | **ElasticNet** | `alpha`, `l1_ratio` |
| 5 | **Support Vector Regression (SVR)** | `C`, `epsilon`, `kernel` |
| 6 | **k-Nearest Neighbors (KNN)** | `n_neighbors`, `metric` |
| 7 | **Decision Tree** | `max_depth`, `min_samples_leaf` |
| 8 | **Random Forest** | `n_estimators`, `max_features` |
| 9 | **Extra Trees** | `n_estimators`, `max_features` |
| 10 | **Gradient Boosting (sklearn)** | `n_estimators`, `learning_rate` |
| 11 | **AdaBoost** | `n_estimators`, `learning_rate` |
| 12 | **BaggingRegressor** | `n_estimators` |

### Advanced Gradient Boosting

| # | Model | Notes |
|---|-------|-------|
| 13 | **XGBoost** | GPU-capable; `n_estimators`, `max_depth`, `eta`, `subsample` |
| 14 | **LightGBM** | Histogram-based; fast on large datasets; `num_leaves`, `min_child_samples` |
| 15 | **CatBoost** | Native categorical handling; `depth`, `iterations`, `learning_rate` |
| 16 | **HistGradientBoosting** | sklearn implementation; fast on large data |

### Deep Learning

| # | Model | Architecture | Framework |
|---|-------|-------------|-----------|
| 17 | **MLP (sklearn)** | Fully connected layers | scikit-learn |
| 18 | **TabNet** | Attentive sequential feature selection | pytorch-tabnet |
| 19 | **PyTorch MLP** | Custom 4-layer network with BatchNorm + Dropout | PyTorch |
| 20 | **1D CNN** | Convolutional layers over fingerprint bits | PyTorch |

### Ensemble / Meta-Models

| # | Model | Strategy |
|---|-------|---------|
| 21 | **VotingRegressor** | Average of RF + XGBoost + LightGBM |
| 22 | **StackingRegressor** | RF + CatBoost + LightGBM → Ridge meta-learner |

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         QSAR ML Pipeline                                │
│                                                                         │
│  [1. DATA COLLECTION]                                                   │
│   ChEMBL API → raw IC50 data (SMILES + bioactivity)                   │
│                     │                                                   │
│  [2. PREPROCESSING]                                                     │
│   Filter → Deduplicate → Unit Standardize → IC50→pIC50 conversion     │
│                     │                                                   │
│  [3. FEATURIZATION]                                                     │
│   SMILES → Morgan FP (2048-bit) + RDKit Descriptors (200+)            │
│                     │                                                   │
│  [4. FEATURE ENGINEERING]                                               │
│   Variance Threshold Filter → Correlation Filter → StandardScaler      │
│                     │                                                   │
│  [5. TRAIN/TEST SPLIT]                                                  │
│   Scaffold Split (Butina clustering) → 80% train / 20% test           │
│                     │                                                   │
│  [6. MODEL TRAINING]                                                    │
│   20+ models × 5-fold cross-validation                                │
│   + Optuna hyperparameter optimization (for top models)                │
│                     │                                                   │
│  [7. EVALUATION]                                                        │
│   R², RMSE, MAE on held-out test set → Benchmark Table                │
│                     │                                                   │
│  [8. EXPLAINABILITY]                                                    │
│   SHAP values → Feature importance → Molecular highlights             │
│                     │                                                   │
│  [9. MODEL PERSISTENCE]                                                 │
│   joblib/pickle → .pkl files; PyTorch → .pt checkpoints               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

##  Quick Start

### Option A: Run in Google Colab (Recommended for Beginners)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/QSAR-ML-Drug-Discovery/blob/main/notebooks/QSAR_ML_Models.ipynb)

Click the badge above — all dependencies install automatically in the first cell.

### Option B: Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/QSAR-ML-Drug-Discovery.git
cd QSAR-ML-Drug-Discovery

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate qsar-ml

# 3. Install the src package in editable mode
pip install -e .

# 4. Launch Jupyter
jupyter lab notebooks/QSAR_ML_Models.ipynb
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Conda (Anaconda or Miniconda) — recommended for RDKit

### Step-by-step

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/QSAR-ML-Drug-Discovery.git
cd QSAR-ML-Drug-Discovery

# Create conda environment with RDKit (pip RDKit also works since 2022)
conda create -n qsar-ml python=3.10
conda activate qsar-ml

# Install all dependencies
pip install -r requirements.txt

# (Optional) Install as a local package for importing src modules
pip install -e .
```

### Verify Installation

```python
python -c "
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import xgboost, lightgbm, catboost
from pytorch_tabnet.tab_model import TabNetRegressor
print('All dependencies installed successfully!')
"
```

---

##  Usage Guide

### 1. Fetch Data from ChEMBL

```python
from src.data.fetch_chembl import fetch_bioactivity_data

# Fetch IC50 data for Acetylcholinesterase (CHEMBL220)
df = fetch_bioactivity_data(
    target_chembl_id="CHEMBL220",
    bioactivity_type="IC50",
    output_path="data/raw/acetylcholinesterase_raw.csv"
)
print(f"Fetched {len(df)} bioactivity records")
```

### 2. Preprocess Data

```python
from src.data.preprocess import preprocess_bioactivity

df_clean = preprocess_bioactivity(
    df,
    pic50_min=3.0,
    pic50_max=12.0,
    output_path="data/processed/clean_dataset.csv"
)
print(f"Cleaned dataset: {len(df_clean)} compounds")
```

### 3. Generate Molecular Features

```python
from src.features.fingerprints import compute_morgan_fingerprints
from src.features.descriptors import compute_rdkit_descriptors

# Morgan fingerprints (2048-bit, radius 2)
X_fp = compute_morgan_fingerprints(
    smiles_list=df_clean["canonical_smiles"].tolist(),
    radius=2,
    n_bits=2048
)

# RDKit 2D physicochemical descriptors
X_desc = compute_rdkit_descriptors(
    smiles_list=df_clean["canonical_smiles"].tolist()
)

# Target variable
y = df_clean["pIC50"].values
```

### 4. Train All Models

```python
from src.models.train import run_benchmark

results = run_benchmark(
    X=X_fp,
    y=y,
    test_size=0.2,
    cv_folds=5,
    random_state=42
)
# results → pd.DataFrame with R², RMSE, MAE for each model
```

### 5. Generate SHAP Explanations

```python
from src.visualization.results_plots import plot_shap_summary

# For the best model (e.g., XGBoost)
plot_shap_summary(
    model=results["best_model"],
    X_test=X_test,
    feature_names=feature_names,
    save_path="results/figures/shap_summary.png"
)
```

---

##  Results & Benchmarks

*Example benchmark results on the AChE dataset with Morgan fingerprints (2048-bit, radius=2). Your results may vary depending on the dataset version and train/test split.*

| Rank | Model | R² (Test) | RMSE | MAE | CV R² (mean ± std) |
|------|-------|-----------|------|-----|-------------------|
| 🥇 1 | **CatBoost** | 0.742 | 0.721 | 0.543 | 0.729 ± 0.031 |
| 🥈 2 | **LightGBM** | 0.738 | 0.726 | 0.548 | 0.725 ± 0.029 |
| 🥉 3 | **XGBoost** | 0.731 | 0.737 | 0.556 | 0.718 ± 0.034 |
| 4 | StackingRegressor | 0.729 | 0.740 | 0.559 | 0.715 ± 0.028 |
| 5 | Random Forest | 0.718 | 0.754 | 0.571 | 0.701 ± 0.035 |
| 6 | TabNet | 0.711 | 0.762 | 0.580 | 0.695 ± 0.041 |
| 7 | PyTorch MLP | 0.705 | 0.770 | 0.588 | 0.688 ± 0.038 |
| 8 | Extra Trees | 0.698 | 0.779 | 0.596 | 0.681 ± 0.040 |
| 9 | SVR (RBF) | 0.692 | 0.787 | 0.601 | 0.676 ± 0.036 |
| 10 | MLP (sklearn) | 0.681 | 0.800 | 0.617 | 0.665 ± 0.044 |
| ... | ... | ... | ... | ... | ... |

> 📝 **Note**: These are example values. Run the notebook to reproduce exact results on the current ChEMBL dataset.

---

##  Explainability with SHAP

SHAP (SHapley Additive exPlanations) assigns each feature an importance score for each prediction.

```python
import shap
import xgboost as xgb

# Train XGBoost model
model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05)
model.fit(X_train, y_train)

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Waterfall plot for one prediction
shap.waterfall_plot(explainer(X_test)[0])
```

SHAP allows you to answer:
- **Which molecular descriptors most strongly predict high pIC50?**
- **How does LogP influence the predicted potency for each compound?**
- **Why did the model predict a low pIC50 for compound X?**

---

## TabNet: Why Attentive Tabular Learning Matters

TabNet is a deep learning architecture specifically designed for tabular data. Unlike standard MLPs, TabNet uses **sequential attention** to select which features to use at each decision step.

```
Input Features
      │
      ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Step 1:    │     │  Step 2:    │     │  Step 3:    │
│  Attention  │────▶│  Attention  │────▶│  Attention  │
│  Mask (M₁)  │     │  Mask (M₂)  │     │  Mask (M₃)  │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      ▼                   ▼                   ▼
   Transform           Transform           Transform
      │                   │                   │
      └───────────────────┴───────────────────┘
                          │
                          ▼
                    Final Prediction
```

Key advantages:
- **Built-in feature selection** — learns which features to attend to
- **Interpretable** — attention masks reveal feature importance per step
- **No preprocessing required** — works directly on raw tabular features

---

##  Background: The Drug Discovery Context

```
Target Identification → Hit Discovery → Lead Optimization → Preclinical → Clinical Trials
                              ↑
                        QSAR Models
                    (Virtual Screening)
```

QSAR models are used in the **Hit Discovery** and **Lead Optimization** phases to:

1. **Virtual screen** large compound libraries (e.g., ZINC: 750M compounds)
2. **Predict ADMET** properties (absorption, distribution, metabolism, excretion, toxicity)
3. **Guide medicinal chemistry** — identify which structural modifications improve potency
4. **Build SAR (Structure-Activity Relationship) maps** — understand what parts of the molecule drive activity

---

## Contributing

Contributions are welcome! Please see the workflow below:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-model`
3. Make your changes and add tests
4. Run tests: `pytest tests/`
5. Submit a Pull Request

### Ideas for Contributions

- Add new models (GraphNets, Transformers, GNNs)
- Add more featurization methods (Mordred descriptors, 3D conformer-based)
- Add classification support (active/inactive binary prediction)
- Add Streamlit/Gradio web interface for real-time prediction
- Add more ChEMBL target examples

---


---

## 📖 Further Reading

- Cherkasov et al. (2014). QSAR Modeling: Where Have You Been? Where Are You Going To? *J. Med. Chem.* [DOI](https://doi.org/10.1021/jm4004285)
- Rogers & Hahn (2010). Extended-Connectivity Fingerprints. *J. Chem. Inf. Model.* [DOI](https://doi.org/10.1021/ci100050t)
- Arik & Pfister (2021). TabNet: Attentive Interpretable Tabular Learning. *AAAI.* [arXiv](https://arxiv.org/abs/1908.07442)
- Gaulton et al. (2017). The ChEMBL database in 2017. *Nucleic Acids Res.* [DOI](https://doi.org/10.1093/nar/gkw1074)
- Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions (SHAP). *NeurIPS.* [arXiv](https://arxiv.org/abs/1705.07874)

---

##  License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

##  Community & Policies

This repository follows standard open-source governance practices:

| Document | Purpose |
|----------|---------|
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to set up your dev environment and submit changes |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community behavior expectations (Contributor Covenant) |
| [SECURITY.md](SECURITY.md) | How to report vulnerabilities responsibly |
| [CHANGELOG.md](CHANGELOG.md) | Version history, following Keep a Changelog |
| [CITATION.cff](CITATION.cff) | Machine-readable citation metadata (powers GitHub's "Cite this repository" button) |
| [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) | Checklist for PR submissions |
| [.github/ISSUE_TEMPLATE/](.github/ISSUE_TEMPLATE/) | Structured bug report / feature request / question forms |
| [.github/dependabot.yml](.github/dependabot.yml) | Automated weekly dependency update PRs |
| [.github/CODEOWNERS](.github/CODEOWNERS) | Automatic reviewer assignment on PRs |

Common development tasks are wrapped in the [Makefile](Makefile):

```bash
make install-dev   # editable install + dev dependencies
make test           # run unit tests
make lint            # flake8 + isort + black --check
make format          # auto-format with black + isort
make notebook        # launch Jupyter Lab on the main notebook
make benchmark        # run the 20+ model CLI benchmark
make clean            # remove caches and build artifacts
```

---

| Resource | Link |
|----------|------|
| ChEMBL Database | [EMBL-EBI](https://www.ebi.ac.uk/chembl/) |
| RDKit Documentation | [rdkit.org](https://www.rdkit.org/docs/) |
| pytorch-tabnet | [GitHub](https://github.com/dreamquark-ai/tabnet) |
| SHAP Documentation | [shap.readthedocs.io](https://shap.readthedocs.io/) |
| Optuna | [optuna.org](https://optuna.org/) |


---
