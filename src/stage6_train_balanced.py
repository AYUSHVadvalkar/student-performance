import numpy as np
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


print("=" * 60)
print("STAGE 6 - BALANCED MLP MODEL TRAINING")
print("=" * 60)


# ------------------------------------------------------------
# Load processed training data
# ------------------------------------------------------------

print("\nLoading training data...")

X_train = np.load("model/X_train.npy")
y_train = np.load("model/y_train.npy")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# ------------------------------------------------------------
# Check class distribution
# ------------------------------------------------------------

fail_count = np.sum(y_train == 0)
pass_count = np.sum(y_train == 1)

print("\nTraining class distribution:")
print("FAIL:", fail_count)
print("PASS:", pass_count)


# ------------------------------------------------------------
# Create sample weights
# ------------------------------------------------------------

# Give more importance to the minority FAIL class
fail_weight = pass_count / fail_count

sample_weights = np.where(
    y_train == 0,
    fail_weight,
    1.0
)

print("\nSample weights:")
print("FAIL weight:", round(fail_weight, 2))
print("PASS weight: 1.00")


# ------------------------------------------------------------
# Create MLP model
# ------------------------------------------------------------

print("\nCreating balanced MLP model...")

mlp_balanced = MLPClassifier(
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


# ------------------------------------------------------------
# Train model
# ------------------------------------------------------------

print("\nTraining balanced MLP...")

mlp_balanced.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)

print("Training completed!")


# ------------------------------------------------------------
# Training accuracy
# ------------------------------------------------------------

train_pred = mlp_balanced.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    train_pred
)

print("\nTraining Accuracy:")
print(f"{train_accuracy * 100:.2f}%")


# ------------------------------------------------------------
# Save improved model
# ------------------------------------------------------------

model_path = "model/mlp_student_performance_balanced.pkl"

joblib.dump(
    mlp_balanced,
    model_path
)

print("\nBalanced model saved successfully:")
print(model_path)


print("\n" + "=" * 60)
print("STAGE 6.1 COMPLETED!")
print("=" * 60)