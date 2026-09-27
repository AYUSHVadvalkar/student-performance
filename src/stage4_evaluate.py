import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("=" * 60)
print("STAGE 4 - MLP MODEL EVALUATION")
print("=" * 60)

print("\nLoading test data...")

X_test = np.load("model/X_test.npy")
y_test = np.load("model/y_test.npy")

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print("\nLoading trained MLP model...")

mlp = joblib.load("model/mlp_student_performance.pkl")

print("Model loaded successfully!")

print("\nMaking predictions...")

y_pred = mlp.predict(X_test)

print("Predictions completed!")

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

# Classification report
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

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

print("\nConfusion Matrix Explanation:")

print("True FAIL predicted as FAIL :", cm[0][0])
print("True FAIL predicted as PASS :", cm[0][1])
print("True PASS predicted as FAIL :", cm[1][0])
print("True PASS predicted as PASS :", cm[1][1])

# Plot confusion matrix
plt.figure(figsize=(7, 5))

plt.imshow(cm, interpolation="nearest")

plt.title("MLP Student Performance - Confusion Matrix")

plt.colorbar()

plt.xticks([0, 1], ["FAIL", "PASS"])
plt.yticks([0, 1], ["FAIL", "PASS"])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "graphs/confusion_matrix.png",
    dpi=300
)

plt.show()

# Save predictions
np.save("model/y_test_pred.npy", y_pred)

print("\nPrediction file saved:")
print("model/y_test_pred.npy")

print("\n" + "=" * 60)
print("STAGE 4 COMPLETED!")
print("=" * 60)