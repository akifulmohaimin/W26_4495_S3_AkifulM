import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("heart_project/heart.csv")

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nClass distribution:")
print(df["HeartDisease"].value_counts(normalize=True))

# -----------------------------
# 2. BASIC CLEANING
# -----------------------------
# Some versions of this dataset contain 0 values in columns like Cholesterol or RestingBP
# that are not physiologically meaningful. Treat them as missing.
for col in ["Cholesterol", "RestingBP", "MaxHR"]:
    df[col] = df[col].replace(0, np.nan)

# Optional: inspect missing values
print("\nMissing values after zero-cleaning:")
print(df.isna().sum())

# -----------------------------
# 3. FEATURES AND TARGET
# -----------------------------
target_col = "HeartDisease"

categorical_features = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope"
]

numeric_features = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak"
]

X = df[categorical_features + numeric_features]
y = df[target_col]

# -----------------------------
# 4. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 5. PREPROCESSORS
# -----------------------------
numeric_transformer_lr = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

numeric_transformer_rf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor_lr = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_lr, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

preprocessor_rf = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_rf, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# 6. PIPELINES
# -----------------------------
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_lr),
    ("model", LogisticRegression(max_iter=3000, random_state=42))
])

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_rf),
    ("model", RandomForestClassifier(random_state=42))
])

# -----------------------------
# 7. BASELINE MODELS
# -----------------------------
lr_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

pred_lr = lr_pipeline.predict(X_test)
pred_rf = rf_pipeline.predict(X_test)

prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]
prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

print("\n" + "=" * 70)
print("BASELINE MODEL RESULTS")
print("=" * 70)

print("\nLOGISTIC REGRESSION")
print("Accuracy:", round(accuracy_score(y_test, pred_lr), 4))
print("ROC-AUC :", round(roc_auc_score(y_test, prob_lr), 4))
print(classification_report(y_test, pred_lr))

print("\nRANDOM FOREST")
print("Accuracy:", round(accuracy_score(y_test, pred_rf), 4))
print("ROC-AUC :", round(roc_auc_score(y_test, prob_rf), 4))
print(classification_report(y_test, pred_rf))

# -----------------------------
# 8. CROSS-VALIDATION + TUNING
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__solver": ["liblinear", "lbfgs"]
}

rf_param_grid = {
    "model__n_estimators": [200, 300, 400],
    "model__max_depth": [4, 6, 8, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

lr_grid = GridSearchCV(
    estimator=lr_pipeline,
    param_grid=lr_param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

lr_grid.fit(X_train, y_train)
rf_grid.fit(X_train, y_train)

best_lr = lr_grid.best_estimator_
best_rf = rf_grid.best_estimator_

print("\n" + "=" * 70)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 70)

print("\nBest Logistic Regression Params:", lr_grid.best_params_)
print("Best Logistic Regression CV ROC-AUC:", round(lr_grid.best_score_, 4))

print("\nBest Random Forest Params:", rf_grid.best_params_)
print("Best Random Forest CV ROC-AUC:", round(rf_grid.best_score_, 4))

# -----------------------------
# 9. FINAL EVALUATION
# -----------------------------
models = {
    "Tuned Logistic Regression": best_lr,
    "Tuned Random Forest": best_rf
}

results = []

for name, model in models.items():
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test, prob)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "ROC_AUC": round(roc, 4)
    })

    print("\n" + "=" * 70)
    print(name.upper())
    print("=" * 70)
    print("Accuracy:", round(acc, 4))
    print("ROC-AUC :", round(roc, 4))
    print(classification_report(y_test, pred))

results_df = pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False)
print("\nMODEL COMPARISON")
print(results_df)

best_model_name = results_df.iloc[0]["Model"]
best_model = best_lr if best_model_name == "Tuned Logistic Regression" else best_rf

print(f"\nBest model selected: {best_model_name}")

# -----------------------------
# 10. CONFUSION MATRIX
# -----------------------------
best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_pred)

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.savefig("heart_best_confusion_matrix.png")
plt.show()

# -----------------------------
# 11. ROC CURVE
# -----------------------------
best_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, best_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"{best_model_name} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {best_model_name}")
plt.legend()
plt.tight_layout()
plt.savefig("heart_best_roc_curve.png")
plt.show()

# -----------------------------
# 12. FEATURE IMPORTANCE / COEFFICIENTS
# -----------------------------
# Get transformed feature names
preprocessor = best_model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()

if "Random Forest" in best_model_name:
    importances = best_model.named_steps["model"].feature_importances_
    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print("\nTop Feature Importances:")
    print(feature_df.head(15))

    plt.figure(figsize=(10, 8))
    plt.barh(feature_df.head(15)["Feature"][::-1], feature_df.head(15)["Importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig("heart_feature_importance.png")
    plt.show()

else:
    coefficients = best_model.named_steps["model"].coef_[0]
    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "AbsCoefficient": np.abs(coefficients)
    }).sort_values(by="AbsCoefficient", ascending=False)

    print("\nTop Logistic Regression Coefficients:")
    print(feature_df.head(15))

    plt.figure(figsize=(10, 8))
    top_coef = feature_df.head(15).sort_values(by="Coefficient")
    plt.barh(top_coef["Feature"], top_coef["Coefficient"])
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.title("Top 15 Logistic Regression Coefficients")
    plt.tight_layout()
    plt.savefig("heart_feature_coefficients.png")
    plt.show()

feature_df.to_csv("heart_model_feature_analysis.csv", index=False)

# -----------------------------
# 13. SAVE BEST MODEL
# -----------------------------
joblib.dump(best_model, "best_heart_model_pipeline.pkl")
joblib.dump(categorical_features, "heart_categorical_features.pkl")
joblib.dump(numeric_features, "heart_numeric_features.pkl")

print("\nSaved files:")
print("- best_heart_model_pipeline.pkl")
print("- heart_categorical_features.pkl")
print("- heart_numeric_features.pkl")
print("- heart_best_confusion_matrix.png")
print("- heart_best_roc_curve.png")
print("- heart_model_feature_analysis.csv")
