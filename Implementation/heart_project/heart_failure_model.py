import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("heart.csv")

# -----------------------------
# Encode categorical columns
# -----------------------------
categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

encoders = {}
for col in categorical:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -----------------------------
# Features and target
# -----------------------------
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Logistic Regression
# -----------------------------
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)
prob_lr = lr.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)
print("Accuracy:", accuracy_score(y_test, pred_lr))
print(classification_report(y_test, pred_lr))

# -----------------------------
# Random Forest
# -----------------------------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
prob_rf = rf.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("RANDOM FOREST")
print("=" * 50)
print("Accuracy:", accuracy_score(y_test, pred_rf))
print(classification_report(y_test, pred_rf))

# -----------------------------
# Confusion Matrices
# -----------------------------
cm_lr = confusion_matrix(y_test, pred_lr)
cm_rf = confusion_matrix(y_test, pred_rf)

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lr)
disp.plot(ax=ax)
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig("lr_confusion_matrix.png")
plt.show()

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
disp.plot(ax=ax)
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.savefig("rf_confusion_matrix.png")
plt.show()

# -----------------------------
# ROC Curves
# -----------------------------
fpr_lr, tpr_lr, _ = roc_curve(y_test, prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(7, 5))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_lr:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_comparison.png")
plt.show()

# -----------------------------
# Feature Importance (Random Forest)
# -----------------------------
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop Feature Importances (Random Forest):")
print(feature_importance)

feature_importance.to_csv("rf_feature_importance.csv", index=False)

plt.figure(figsize=(8, 6))
plt.barh(feature_importance["Feature"], feature_importance["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
plt.show()