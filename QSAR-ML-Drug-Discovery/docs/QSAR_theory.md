# QSAR Theory: Background & Further Reading

This document supplements the main [README.md](../README.md) with deeper theoretical background
for those wanting to understand the *why* behind each modeling decision in this repository.

## 1. The QSAR Hypothesis

The foundational assumption of QSAR is that **structurally similar molecules tend to exhibit
similar biological activity** — sometimes called the "similar property principle." This isn't
universally true (activity cliffs exist, where tiny structural changes cause large activity
shifts), but it holds well enough on average to make statistical learning tractable.

```
Structure (X) ──f(X)──▶ Activity (Y)
```

QSAR modeling is the process of learning `f` from a training set of compounds with known
structure (X) and measured activity (Y), then using the learned `f` to predict Y for new,
untested compounds.

## 2. Why pIC50 and not raw IC50?

IC50 values are inherently a *ratio* measurement (the concentration needed for 50% inhibition),
and most real-world bioactivity datasets span 3–6 orders of magnitude (e.g., 0.5 nM to 500,000 nM).
Regressing directly on IC50:

- Heavily weights the loss function toward the high-IC50 (weak/inactive) compounds
- Violates the homoscedasticity assumption of many regression models
- Produces nonsensical negative predictions for highly potent compounds if extrapolated

The negative log transform compresses this into a roughly Gaussian-shaped variable centered
around pIC50 = 5–7 for most bioactivity datasets, which behaves much better under squared-error
loss functions (used by virtually every regressor in this repository).

## 3. Choice of Molecular Representation

| Representation | Captures | Limitations |
|---|---|---|
| Morgan/ECFP fingerprints | Local substructure environments | No global 3D shape; bit collisions possible |
| RDKit 2D descriptors | Global physicochemical properties | Loses fine-grained substructure detail |
| MACCS keys | Presence of specific known motifs | Fixed, coarse vocabulary (166 keys) |
| 3D descriptors (not used here) | Conformational/steric effects | Requires conformer generation; computationally expensive |
| Graph neural network embeddings (not used here) | Learned, task-specific representations | Requires more data and tuning; less interpretable |

This repository deliberately starts with the two most standard, computationally cheap, and
widely benchmarked representations (Morgan FP + RDKit descriptors) because they remain
extremely competitive baselines in the QSAR literature, and because they let learners focus on
the modeling methodology rather than representation engineering.

## 4. Why Benchmark 20+ Models?

There is no universal "best" QSAR algorithm — performance depends heavily on dataset size,
feature representation, and noise level in the bioactivity measurements. Benchmarking broadly
across linear, kernel, tree-ensemble, and neural architectures:

- Reveals whether the problem is closer to linear or highly nonlinear
- Surfaces whether tree-based models (which handle sparse binary fingerprints well) outperform
  distance-based models (which often struggle in high-dimensional binary spaces)
- Provides a robust baseline against which more exotic architectures (GNNs, transformers) should
  be compared before adopting additional complexity

In practice, for fingerprint-based QSAR tasks, gradient-boosted trees (XGBoost, LightGBM, CatBoost)
and random forests are consistently strong performers, while attention-based tabular models like
TabNet are competitive but more sensitive to hyperparameter tuning and dataset size.

## 5. Cross-Validation Strategy

This repository uses standard **random K-fold cross-validation**. For production QSAR work,
consider **scaffold-based splitting** (e.g., via Bemis-Murcko scaffolds or Butina clustering),
which holds out entire chemical series rather than individual compounds. Random splits tend to
overestimate generalization performance because structurally near-identical analogs of a training
compound often end up in the test set, inflating apparent accuracy.

## 6. SHAP for QSAR Interpretability

SHAP values decompose a prediction into additive per-feature contributions, grounded in
cooperative game theory (Shapley values). For tree-based QSAR models, `shap.TreeExplainer`
computes *exact* SHAP values efficiently (polynomial time rather than exponential), making it
practical even for thousands of compounds.

When applied to RDKit descriptors, SHAP plots typically reveal chemically sensible patterns —
for example, increasing LogP (lipophilicity) often correlates positively with potency against
intracellular targets up to a point, after which solubility and off-target effects dominate.

## Key References

- Cherkasov, A. et al. (2014). QSAR Modeling: Where Have You Been? Where Are You Going To?
  *Journal of Medicinal Chemistry*, 57(12), 4977–5010.
- Rogers, D. & Hahn, M. (2010). Extended-Connectivity Fingerprints.
  *Journal of Chemical Information and Modeling*, 50(5), 742–754.
- Arik, S.O. & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. *AAAI*.
- Lundberg, S.M. & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- Gaulton, A. et al. (2017). The ChEMBL database in 2017. *Nucleic Acids Research*, 45(D1), D945–D954.
