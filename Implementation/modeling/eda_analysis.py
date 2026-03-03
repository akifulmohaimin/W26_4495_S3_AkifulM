import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import load_and_clean_data, METRIC_DIR, PLOTS_DIR

FEATURES_4 = ["Glucose", "BMI", "BloodPressure", "Age"]

def save_summary_tables(df: pd.DataFrame):
    # Summary statistics
    summary = df.describe(include="all").T
    summary_path = os.path.join(METRIC_DIR, "eda_summary_statistics.csv")
    summary.to_csv(summary_path)

    # Missing values check 
    missing = df.isna().sum().to_frame("missing_count")
    missing_path = os.path.join(METRIC_DIR, "eda_missing_values.csv")
    missing.to_csv(missing_path)

    print("Saved:", summary_path)
    print("Saved:", missing_path)


def plot_histograms(df: pd.DataFrame, cols):
    for c in cols:
        plt.figure()
        plt.hist(df[c], bins=30)
        plt.title(f"Distribution of {c}")
        plt.xlabel(c)
        plt.ylabel("Count")
        out = os.path.join(PLOTS_DIR, f"{c}_distribution.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print("Saved:", out)


def plot_boxplots(df: pd.DataFrame, cols):
    for c in cols:
        plt.figure()
        plt.boxplot(df[c].values, vert=False)
        plt.title(f"Outlier Check (Boxplot): {c}")
        plt.xlabel(c)
        out = os.path.join(PLOTS_DIR, f"{c}_boxplot.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print("Saved:", out)


def plot_correlation_heatmap(df: pd.DataFrame, cols):
    # Correlation among selected columns + Outcome
    corr_df = df[cols + ["Outcome"]].corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr_df.values)
    plt.title("Correlation Heatmap (Selected Features + Outcome)")
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr_df.index)), corr_df.index)

    # Add correlation values
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            plt.text(j, i, f"{corr_df.values[i, j]:.2f}", ha="center", va="center")

    plt.colorbar()
    out = os.path.join(PLOTS_DIR, "correlation_heatmap_selected.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

    # Save correlation matrix as CSV 
    corr_csv = os.path.join(METRIC_DIR, "correlation_matrix_selected.csv")
    corr_df.to_csv(corr_csv)
    print("Saved:", corr_csv)

    return corr_df


def outlier_report_iqr(df: pd.DataFrame, cols):
    
    #outlier check using IQR rule.
    rows = []
    for c in cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((df[c] < lower) | (df[c] > upper)).sum()
        rows.append({
            "feature": c,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outlier_count": int(outliers),
            "outlier_percent": float(outliers / len(df) * 100.0)
        })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(METRIC_DIR, "outlier_report_iqr.csv")
    out_df.to_csv(out_path, index=False)
    print("Saved:", out_path)


def feature_selection_justification(corr_df: pd.DataFrame):
    
    corr_with_outcome = corr_df["Outcome"].drop("Outcome").sort_values(ascending=False)

    report = {
        "selected_features": FEATURES_4,
        "justification_points": [
            "Selected features are clinically meaningful and commonly available in patient-accessible records.",
            "Selected features are feasible to extract from PDFs/OCR or accept as user input.",
            "Correlation with Outcome was assessed to support relevance for risk modeling."
        ],
        "correlation_with_outcome": {k: float(v) for k, v in corr_with_outcome.to_dict().items()}
    }

    out_path = os.path.join(METRIC_DIR, "feature_selection_justification.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved:", out_path)
    print("\nCorrelation with Outcome (descending):")
    print(corr_with_outcome)


def main():
    df = load_and_clean_data()

    # 1) summary statistics + missing values check
    save_summary_tables(df)

    # 2) Distribution plots
    plot_histograms(df, FEATURES_4)

    # 3) Outlier visuals + IQR report
    plot_boxplots(df, FEATURES_4)
    outlier_report_iqr(df, FEATURES_4)

    # 4) Correlation matrix + heatmap
    corr_df = plot_correlation_heatmap(df, FEATURES_4)

    # 5) Feature selection justification artifact
    feature_selection_justification(corr_df)

    print("\nEDA + Feature Validation completed successfully.")
    print("Check Implementation/outputs/plots and Implementation/outputs/metrics")


if __name__ == "__main__":
    main()