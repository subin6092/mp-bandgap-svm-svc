# MP Band Gap Prediction (Two-Stage SVM, Magpie Features)

This repo reproduces a classic materials-ML baseline:
(Ya Zhuo, Aria Mansouri Tehrani, and Jakoah Brgoch
The Journal of Physical Chemistry Letters 2018 9 (7), 1668-1673
DOI: 10.1021/acs.jpclett.8b00124)

1) **SVC (RBF kernel)** to classify **metal vs nonmetal**
2) **SVR (RBF kernel)** to predict **experimental band gap** for **nonmetals**

Features are **composition-only** descriptors (Magpie-style) generated with **matminer**.

## Dataset (from matminer)
- `matbench_expt_is_metal` (classification)
- `matbench_expt_gap` (regression)

Loaded via `matminer.datasets.load_dataset`.

## Features
Generated from chemical formula using:
- `Stoichiometry()`
- `ElementProperty.from_preset("magpie", impute_nan=True)`
- `ValenceOrbital()`

## Models
- Classification: `SVC(kernel="rbf", class_weight="balanced")`
- Regression: `SVR(kernel="rbf")`

Both models use:
- `SimpleImputer(strategy="median")`
- `StandardScaler()`
inside a `Pipeline` to prevent data leakage.

Hyperparameters tuned with `GridSearchCV` and 5-fold CV.
