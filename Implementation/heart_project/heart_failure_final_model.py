import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("heart.csv")

# -----------------------------
# 2. ENCODE CATEGORICAL COLUMNS
# -----------------------------
categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

encoders = {}
for col in categorical:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------
# 3. CORRELATION HEATMAP
# -----------------------------
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap - Heart Failure Dataset")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

# -----------------------------
# 4. FEATURES AND TARGET
# -----------------------------
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# -----------------------------
# 5. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 6. BASELINE MODELS
# -----------------------------
lr = LogisticRegression(max_iter=2000, random_state=42)
rf = RandomForestClassifier(random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

pred_lr = lr.predict(X_test)
pred_rf = rf.predict(X_test)

prob_lr = lr.predict_proba(X_test)[:, 1]
prob_rf = rf.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("BASELINE MODEL RESULTS")
print("=" * 60)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, pred_lr))
print(classification_report(y_test, pred_lr))

print("\nRandom Forest Accuracy:", accuracy_score(y_test, pred_rf))
print(classification_report(y_test, pred_rf))

# -----------------------------
# 7. CROSS-VALIDATION
# -----------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lr_cv_scores = cross_val_score(lr, X, y, cv=cv, scoring="accuracy")
rf_cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")

print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)
print("Logistic Regression CV Scores:", lr_cv_scores)
print("Logistic Regression Mean CV Accuracy:", lr_cv_scores.mean())

print("\nRandom Forest CV Scores:", rf_cv_scores)
print("Random Forest Mean CV Accuracy:", rf_cv_scores.mean())

# -----------------------------
# 8. HYPERPARAMETER TUNING
# -----------------------------
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING RESULTS")
print("=" * 60)
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# -----------------------------
# 9. FINAL EVALUATION OF BEST MODEL
# -----------------------------
best_pred = best_rf.predict(X_test)
best_prob = best_rf.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("FINAL TUNED RANDOM FOREST RESULTS")
print("=" * 60)
print("Tuned Random Forest Accuracy:", accuracy_score(y_test, best_pred))
print(classification_report(y_test, best_pred))

# -----------------------------
# 10. CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_test, best_pred)

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=ax)
plt.title("Confusion Matrix - Tuned Random Forest")
plt.tight_layout()
plt.savefig("tuned_rf_confusion_matrix.png")
plt.show()

# -----------------------------
# 11. ROC CURVE
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, best_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Tuned Random Forest (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tuned Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig("tuned_rf_roc_curve.png")
plt.show()

# -----------------------------
# 12. FEATURE IMPORTANCE
# -----------------------------
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop Feature Importances:")
print(feature_importance)

feature_importance.to_csv("tuned_rf_feature_importance.csv", index=False)

plt.figure(figsize=(8, 6))
plt.barh(feature_importance["Feature"], feature_importance["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance - Tuned Random Forest")
plt.tight_layout()
plt.savefig("tuned_rf_feature_importance.png")
plt.show()

# -----------------------------
# 13. SAVE MODEL FOR INTEGRATION
# -----------------------------
joblib.dump(best_rf, "best_heart_rf_model.pkl")
joblib.dump(encoders, "heart_label_encoders.pkl")
joblib.dump(list(X.columns), "heart_feature_columns.pkl")

print("\nSaved files:")
print("- best_heart_rf_model.pkl")
print("- heart_label_encoders.pkl")
print("- heart_feature_columns.pkl")
print("- correlation_heatmap.png")
print("- tuned_rf_confusion_matrix.png")
print("- tuned_rf_roc_curve.png")
print("- tuned_rf_feature_importance.csv")
print("- tuned_rf_feature_importance.png")