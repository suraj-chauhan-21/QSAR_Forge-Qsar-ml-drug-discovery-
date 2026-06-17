# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Graph Neural Network (GNN) baseline using PyTorch Geometric
- Scaffold-based (Bemis-Murcko) train/test splitting
- Streamlit web app for interactive compound screening
- Mordred descriptor support as an alternative to RDKit descriptors
- Binary classification mode (active/inactive) alongside regression

## [1.0.0] — 2025-01-01

### Added
- Initial public release of the QSAR-ML Drug Discovery repository
- ChEMBL data fetching module (`src/data/fetch_chembl.py`)
- Bioactivity preprocessing pipeline with IC50 → pIC50 conversion
  (`src/data/preprocess.py`)
- Morgan fingerprint, MACCS keys, and RDKit fingerprint generation
  (`src/features/fingerprints.py`)
- RDKit 2D physicochemical descriptor computation, including a curated
  Lipinski Rule-of-Five subset (`src/features/descriptors.py`)
- Benchmark training pipeline covering 20+ regression models: linear,
  kernel, tree, ensemble, gradient boosting (XGBoost, LightGBM, CatBoost),
  and neural network models (`src/models/train.py`)
- TabNet attentive tabular learning integration (`src/models/tabnet_model.py`)
- PyTorch MLP and 1D CNN models with sklearn-compatible wrapper
  (`src/models/neural_nets.py`)
- Model evaluation utilities and SHAP explainability plots
  (`src/models/evaluate.py`)
- EDA visualization helpers: pIC50 distribution, correlation heatmaps,
  chemical space (PCA/t-SNE) plots (`src/visualization/eda_plots.py`)
- Companion Jupyter notebook (`notebooks/QSAR_ML_Models.ipynb`) walking
  through the full pipeline end-to-end
- Unit test suite covering preprocessing, featurization, and model training
  (`tests/`)
- GitHub Actions CI workflow (lint + multi-version test matrix + notebook
  smoke test)
- Standard repository scaffolding: README, LICENSE (MIT), CONTRIBUTING,
  CODE_OF_CONDUCT, SECURITY policy, issue/PR templates

[Unreleased]: https://github.com/YOUR_USERNAME/QSAR-ML-Drug-Discovery/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/YOUR_USERNAME/QSAR-ML-Drug-Discovery/releases/tag/v1.0.0
