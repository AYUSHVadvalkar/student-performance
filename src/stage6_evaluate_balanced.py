import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("=" * 60)
print("STAGE 6.2 - BALANCED MLP MODEL EVALUATION")
print("=" * 60)


# ------------------------------------------------------------
# Load test data
# ------------------------------------------------------------

print("\nLoading test data...")

X_test = np.load("model/X_test.npy")
y_test = np.load("model/y_test.npy")

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ------------------------------------------------------------
# Load balanced model
# ------------------------------------------------------------

print("\nLoading balanced MLP model...")

mlp_balanced = joblib.load(
    "model/mlp_student_performance_balanced.pkl"
)

print("Balanced model loaded successfully!")


# ------------------------------------------------------------
# Make predictions
# ------------------------------------------------------------

print("\nMaking predictions...")

y_pred = mlp_balanced.predict(X_test)

print("Predictions completed!")


# ------------------------------------------------------------
# Calculate metrics
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ------------------------------------------------------------
# Display performance
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BALANCED MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")


# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["FAIL", "PASS"],
        zero_division=0
    )
)


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

print("\nConfusion Matrix Explanation:")

print(
    "True FAIL predicted as FAIL :",
    cm[0][0]
)

print(
    "True FAIL predicted as PASS :",
    cm[0][1]
)

print(
    "True PASS predicted as FAIL :",
    cm[1][0]
)

print(
    "True PASS predicted as PASS :",
    cm[1][1]
)


print("\n" + "=" * 60)
print("STAGE 6.2 COMPLETED!")
print("=" * 60)