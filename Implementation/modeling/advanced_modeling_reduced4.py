import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from common import load_and_clean_data, MODEL_DIR, METRIC_DIR, PLOTS_DIR

RANDOM_STATE = 42
FEATURES_4 = ["Glucose", "BMI", "BloodPressure", "Age"]


def metrics_at_threshold(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_json(obj: dict, filename: str):
    path = os.path.join(METRIC_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print("Saved:", path)


def plot_roc(y_true, y_proba, title: str, out_name: str):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()

    out_path = os.path.join(PLOTS_DIR, out_name)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def plot_precision_recall(y_true, y_proba, title: str, out_name: str):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)

    out_path = os.path.join(PLOTS_DIR, out_name)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def main():
    df = load_and_clean_data()

    # Reduced feature set (deployment-aligned)
    for c in FEATURES_4:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in dataset.")

    X = df[FEATURES_4].copy()
    y = df["Outcome"].astype(int)

    # Training/Testing the split for fair evaluation + threshold study
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # -----------------------------
    # Cross-validation (5-fold) – Baseline models
    # -----------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ])

    rf_base = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    lr_cv_auc = cross_val_score(lr_pipe, X, y, cv=cv, scoring="roc_auc")
    rf_cv_auc = cross_val_score(rf_base, X, y, cv=cv, scoring="roc_auc")

    cv_summary = {
        "feature_set": "reduced_4",
        "cv_method": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "logistic_regression": {
            "roc_auc_mean": float(lr_cv_auc.mean()),
            "roc_auc_std": float(lr_cv_auc.std()),
            "fold_scores": [float(v) for v in lr_cv_auc],
        },
        "random_forest_baseline": {
            "roc_auc_mean": float(rf_cv_auc.mean()),
            "roc_auc_std": float(rf_cv_auc.std()),
            "fold_scores": [float(v) for v in rf_cv_auc],
        }
    }
    save_json(cv_summary, "reduced4_cross_validation_summary.json")

    # -----------------------------
    # Part B) Hyperparameter tuning (GridSearchCV) for Random Forest
    # -----------------------------
    param_grid = {
        "n_estimators": [200, 400, 600],
        "max_depth": [None, 4, 6, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    grid = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X_train, y_train)

    best_rf = grid.best_estimator_
    tuning_summary = {
        "feature_set": "reduced_4",
        "best_params": grid.best_params_,
        "best_cv_roc_auc": float(grid.best_score_),
        "note": "GridSearchCV ran on training split only (80%) to avoid test leakage."
    }
    save_json(tuning_summary, "reduced4_rf_tuning_summary.json")

    # Saving the tuned model 

    tuned_model_path = os.path.join(MODEL_DIR, "rf_reduced_4_tuned.pkl")
    joblib.dump(best_rf, tuned_model_path)
    print("Saved:", tuned_model_path)

    # -----------------------------
    # Test-set evaluation + Threshold optimization
    # -----------------------------
    # Fitting baseline RF and tuned RF on train, then evaluating on test

    rf_base.fit(X_train, y_train)
    base_proba = rf_base.predict_proba(X_test)[:, 1]

    best_rf.fit(X_train, y_train)
    tuned_proba = best_rf.predict_proba(X_test)[:, 1]

    thresholds = [0.50, 0.45, 0.40, 0.35, 0.30]

    threshold_rows = []
    for t in thresholds:
        m_base = metrics_at_threshold(y_test.values, base_proba, t)
        m_base["model"] = "RF_baseline"
        threshold_rows.append(m_base)

        m_tuned = metrics_at_threshold(y_test.values, tuned_proba, t)
        m_tuned["model"] = "RF_tuned"
        threshold_rows.append(m_tuned)

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_csv = os.path.join(METRIC_DIR, "reduced4_threshold_analysis.csv")
    threshold_df.to_csv(threshold_csv, index=False)
    print("Saved:", threshold_csv)

    # Saving a compact JSON summary at default threshold 0.5
    test_eval = {
        "RF_baseline_t0.5": metrics_at_threshold(y_test.values, base_proba, 0.50),
        "RF_tuned_t0.5": metrics_at_threshold(y_test.values, tuned_proba, 0.50),
        "note": "Use reduced4_threshold_analysis.csv for recall/precision tradeoffs at different thresholds."
    }
    save_json(test_eval, "reduced4_test_evaluation.json")

    # -----------------------------
    # Visualizations (ROC + PR)
    # -----------------------------

    plot_roc(y_test.values, base_proba, "ROC Curve - RF Baseline (Reduced 4)", "roc_rf_baseline_reduced4.png")
    plot_roc(y_test.values, tuned_proba, "ROC Curve - RF Tuned (Reduced 4)", "roc_rf_tuned_reduced4.png")

    plot_precision_recall(y_test.values, base_proba, "Precision-Recall - RF Baseline (Reduced 4)", "pr_rf_baseline_reduced4.png")
    plot_precision_recall(y_test.values, tuned_proba, "Precision-Recall - RF Tuned (Reduced 4)", "pr_rf_tuned_reduced4.png")

    print("\nAdvanced modeling (Reduced 4) completed.")
    print("Outputs saved to Implementation/outputs/metrics, models, plots.")


if __name__ == "__main__":
    main()