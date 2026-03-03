import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from common import load_and_clean_data, evaluate_model, MODEL_DIR, METRIC_DIR

RANDOM_STATE = 42

def main():
    df = load_and_clean_data()

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = {}

    # Logistic Regression
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
    ])

    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    results["Logistic Regression (8 Features)"] = evaluate_model(
        y_test, y_pred_lr, y_prob_lr
    )

    joblib.dump(lr, os.path.join(MODEL_DIR, "lr_full_8.pkl"))

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    results["Random Forest (8 Features)"] = evaluate_model(
        y_test, y_pred_rf, y_prob_rf
    )

    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_full_8.pkl"))


    with open(os.path.join(METRIC_DIR, "uthara_full_8_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Full 8-feature models trained successfully.")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()