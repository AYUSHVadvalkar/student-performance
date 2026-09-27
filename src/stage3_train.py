import numpy as np
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# STAGE 3 - MLP MODEL TRAINING
# ============================================================

print("=" * 60)
print("STAGE 3 - MLP MODEL TRAINING")
print("=" * 60)


# ============================================================
# 1. LOAD PREPROCESSED DATA
# ============================================================

print("\nLoading preprocessed data...")

X_train = np.load("model/X_train.npy")
X_test = np.load("model/X_test.npy")

y_train = np.load("model/y_train.npy")
y_test = np.load("model/y_test.npy")

print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)


# ============================================================
# 2. CREATE MLP MODEL
# ============================================================

print("\nCreating MLP Neural Network...")

mlp = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=30
)


# ============================================================
# 3. TRAIN MODEL
# ============================================================

print("\nTraining MLP model...")

mlp.fit(X_train, y_train)

print("MLP training completed!")


# ============================================================
# 4. TRAINING PREDICTIONS
# ============================================================

y_train_pred = mlp.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

print("\nTraining Accuracy:")
print(f"{train_accuracy * 100:.2f}%")


# ============================================================
# 5. TESTING PREDICTIONS
# ============================================================

y_test_pred = mlp.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print("\nTesting Accuracy:")
print(f"{test_accuracy * 100:.2f}%")


# ============================================================
# 6. MODEL INFORMATION
# ============================================================

print("\nMODEL INFORMATION")
print("-" * 40)

print("Input features :", X_train.shape[1])
print("Hidden layers  : 2")
print("Hidden neurons : 16, 8")
print("Activation     : ReLU")
print("Optimizer      : Adam")
print("Output         : PASS / FAIL")
print("Iterations     :", mlp.n_iter_)


# ============================================================
# 7. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    mlp,
    "model/mlp_student_performance.pkl"
)

print("\nModel saved successfully:")
print("model/mlp_student_performance.pkl")


# ============================================================
# 8. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("STAGE 3 COMPLETED!")
print("=" * 60)