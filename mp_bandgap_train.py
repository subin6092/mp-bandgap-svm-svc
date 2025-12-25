import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymatgen.core import Composition

from matminer.datasets import load_dataset
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_predict
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


RANDOM_STATE = 42


def ensure_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def featurize_compositions(formulas: pd.Series) -> pd.DataFrame:
    """
    Convert formula strings -> pymatgen Composition -> numeric feature matrix (DataFrame).

    Uses:
      - Stoichiometry()
      - Magpie element-property statistics
      - ValenceOrbital()
    """
    df = pd.DataFrame({"formula": formulas})
    df["composition"] = df["formula"].apply(Composition)

    featurizer = MultipleFeaturizer(
        [
            Stoichiometry(),
            ElementProperty.from_preset("magpie", impute_nan=True),
            ValenceOrbital(),
        ]
    )

    df = featurizer.featurize_dataframe(df, col_id="composition", ignore_errors=True)

    # Keep only numeric features
    feature_cols = [c for c in df.columns if c not in ("formula", "composition")]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return X


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def plot_and_save(fig_path):
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()


def main():
    ensure_dirs()

    run_info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "notes": "Two-stage SVM: SVC (metal vs nonmetal) + SVR (gap for nonmetals), Magpie-like composition features via matminer.",
    }

    # -----------------------------
    # Load datasets
    # -----------------------------
    cls = load_dataset("matbench_expt_is_metal")   # columns include: composition, is_metal
    reg = load_dataset("matbench_expt_gap")        # columns include: composition, gap expt

    # Prepare formulas + labels
    formulas_cls = cls["composition"].astype(str)
    y_cls = cls["is_metal"].astype(int).to_numpy()

    formulas_reg = reg["composition"].astype(str)
    y_reg_all = reg["gap expt"].astype(float).to_numpy()

    # Featurize (composition-only)
    X_cls_df = featurize_compositions(formulas_cls)
    X_reg_df = featurize_compositions(formulas_reg)

    # Filter regression to nonmetals only (gap > 0)
    nonmetal_mask = y_reg_all > 0
    X_reg = X_reg_df.loc[nonmetal_mask].to_numpy()
    y_reg = y_reg_all[nonmetal_mask]
    formulas_reg_nm = formulas_reg.loc[nonmetal_mask].reset_index(drop=True)

    # -----------------------------
    # Splits (keep formulas for error inspection)
    # -----------------------------
    Xc_train, Xc_test, yc_train, yc_test, fc_train, fc_test = train_test_split(
        X_cls_df.to_numpy(),
        y_cls,
        formulas_cls.reset_index(drop=True),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_cls,
    )

    Xr_train, Xr_test, yr_train, yr_test, fr_train, fr_test = train_test_split(
        X_reg,
        y_reg,
        formulas_reg_nm,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # -----------------------------
    # Shared preprocessing
    # -----------------------------
    preprocess = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    # -----------------------------
    # Stage 1: SVC
    # -----------------------------
    svc_pipe = Pipeline(preprocess + [("svc", SVC(kernel="rbf", class_weight="balanced"))])

    svc_grid = {
        "svc__C": [0.5, 1, 2, 5, 10, 50, 100],
        "svc__gamma": ["scale", 0.01, 0.05, 0.1, 0.2],
    }

    svc_search = GridSearchCV(
        svc_pipe,
        svc_grid,
        cv=5,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
    )
    svc_search.fit(Xc_train, yc_train)
    best_svc = svc_search.best_estimator_

    y_pred_cls = best_svc.predict(Xc_test)

    svc_metrics = {
        "best_params": svc_search.best_params_,
        "accuracy": float(accuracy_score(yc_test, y_pred_cls)),
        "balanced_accuracy": float(balanced_accuracy_score(yc_test, y_pred_cls)),
        "precision_metal_1": float(precision_score(yc_test, y_pred_cls, pos_label=1)),
        "recall_metal_1": float(recall_score(yc_test, y_pred_cls, pos_label=1)),
        "f1_metal_1": float(f1_score(yc_test, y_pred_cls, pos_label=1)),
        "confusion_matrix_rows_true_cols_pred": confusion_matrix(yc_test, y_pred_cls).tolist(),
    }

    # ROC-AUC via decision function if available
    svc_scores = None
    if hasattr(best_svc, "decision_function"):
        svc_scores = best_svc.decision_function(Xc_test)
        svc_metrics["roc_auc"] = float(roc_auc_score(yc_test, svc_scores))
    elif hasattr(best_svc, "predict_proba"):
        svc_scores = best_svc.predict_proba(Xc_test)[:, 1]
        svc_metrics["roc_auc"] = float(roc_auc_score(yc_test, svc_scores))

    # Save a text classification report
    cls_report = classification_report(yc_test, y_pred_cls, target_names=["nonmetal(0)", "metal(1)"])
    with open("results/svc_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(cls_report)

    # Plot confusion matrix
    plt.figure()
    ConfusionMatrixDisplay.from_predictions(
        yc_test, y_pred_cls, display_labels=["nonmetal(0)", "metal(1)"]
    )
    plt.title("SVC Confusion Matrix (test set)")
    plot_and_save("figures/svc_confusion_matrix.png")

    # Plot ROC curve if scores exist
    if svc_scores is not None:
        fpr, tpr, _ = roc_curve(yc_test, svc_scores)
        auc = roc_auc_score(yc_test, svc_scores)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"SVC ROC Curve (test set) | AUC={auc:.3f}")
        plot_and_save("figures/svc_roc_curve.png")

    # Save misclassified examples
    mis_idx = np.where(y_pred_cls != yc_test)[0]
    mis_examples = []
    for i in mis_idx[:50]:
        mis_examples.append(
            {"formula": str(fc_test.iloc[i]), "true": int(yc_test[i]), "pred": int(y_pred_cls[i])}
        )
    save_json("results/svc_misclassified_examples.json", mis_examples)

    # -----------------------------
    # Stage 2: SVR
    # -----------------------------
    svr_pipe = Pipeline(preprocess + [("svr", SVR(kernel="rbf"))])

    svr_grid = {
        "svr__C": [1, 2, 5, 10, 20, 50, 100],
        "svr__gamma": ["scale", 0.01, 0.05, 0.1, 0.2],
        "svr__epsilon": [0.01, 0.05, 0.1, 0.2],
    }

    svr_search = GridSearchCV(
        svr_pipe,
        svr_grid,
        cv=5,
        scoring="neg_mean_squared_error",  # old-sklearn-safe; we compute RMSE ourselves
        n_jobs=-1,
        verbose=1,
    )
    svr_search.fit(Xr_train, yr_train)
    best_svr = svr_search.best_estimator_

    y_pred_reg = best_svr.predict(Xr_test)
    rmse, mae, r2 = regression_metrics(yr_test, y_pred_reg)

    # 5-fold CV on training set (honest estimate)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred_cv = cross_val_predict(best_svr, Xr_train, yr_train, cv=cv, n_jobs=-1)
    rmse_cv, mae_cv, r2_cv = regression_metrics(yr_train, y_pred_cv)

    svr_metrics = {
        "best_params": svr_search.best_params_,
        "test_rmse_eV": float(rmse),
        "test_mae_eV": float(mae),
        "test_r2": float(r2),
        "train_cv5_rmse_eV": float(rmse_cv),
        "train_cv5_mae_eV": float(mae_cv),
        "train_cv5_r2": float(r2_cv),
    }

    # Regression plots
    residuals = y_pred_reg - yr_test
    abs_err = np.abs(residuals)

    # Parity plot
    plt.figure()
    plt.scatter(yr_test, y_pred_reg, alpha=0.6)
    minv = min(float(np.min(yr_test)), float(np.min(y_pred_reg)))
    maxv = max(float(np.max(yr_test)), float(np.max(y_pred_reg)))
    plt.plot([minv, maxv], [minv, maxv])
    plt.xlabel("True band gap (eV)")
    plt.ylabel("Predicted band gap (eV)")
    plt.title("SVR Parity Plot (test set)")
    plot_and_save("figures/svr_parity.png")

    # Residuals vs true
    plt.figure()
    plt.scatter(yr_test, residuals, alpha=0.6)
    plt.axhline(0)
    plt.xlabel("True band gap (eV)")
    plt.ylabel("Residual (pred - true) (eV)")
    plt.title("SVR Residuals vs True Gap (test set)")
    plot_and_save("figures/svr_residuals.png")

    # Absolute error histogram
    plt.figure()
    plt.hist(abs_err, bins=30)
    plt.xlabel("|Error| (eV)")
    plt.ylabel("Count")
    plt.title("SVR Absolute Error Distribution (test set)")
    plot_and_save("figures/svr_abs_error_hist.png")

    # Save biggest errors
    worst_idx = np.argsort(-abs_err)[:50]
    worst_examples = []
    for i in worst_idx:
        worst_examples.append(
            {
                "formula": str(fr_test.iloc[i]),
                "true_gap_eV": float(yr_test[i]),
                "pred_gap_eV": float(y_pred_reg[i]),
                "abs_error_eV": float(abs_err[i]),
            }
        )
    save_json("results/svr_worst_examples.json", worst_examples)

    # -----------------------------
    # Save summary
    # -----------------------------
    all_metrics = {
        "run_info": run_info,
        "svc": svc_metrics,
        "svr": svr_metrics,
    }

    save_json("results/metrics.json", all_metrics)
    save_json(
        "results/best_params.json",
        {"svc_best_params": svc_search.best_params_, "svr_best_params": svr_search.best_params_},
    )

    print("\n=== DONE ===")
    print("Saved:")
    print("  results/metrics.json")
    print("  results/best_params.json")
    print("  results/svc_classification_report.txt")
    print("  figures/*.png")
    print("\nKey metrics:")
    print(f"  SVC balanced accuracy: {svc_metrics['balanced_accuracy']:.3f}")
    if "roc_auc" in svc_metrics:
        print(f"  SVC ROC-AUC: {svc_metrics['roc_auc']:.3f}")
    print(f"  SVR RMSE (test): {svr_metrics['test_rmse_eV']:.3f} eV")
    print(f"  SVR MAE  (test): {svr_metrics['test_mae_eV']:.3f} eV")
    print(f"  SVR R^2  (test): {svr_metrics['test_r2']:.3f}")


if __name__ == "__main__":
    main()
