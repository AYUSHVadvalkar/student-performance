import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier

print("=" * 60)
print("STAGE 7.3 - MODEL-BASED FEATURE IMPORTANCE")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load processed test data
# ------------------------------------------------------------

X_test = np.load("model/X_test.npy")
y_test = np.load("model/y_test.npy")

print("\nTest data loaded:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ------------------------------------------------------------
# 2. Load trained MLP model
# ------------------------------------------------------------

import joblib

model = joblib.load(
    "model/mlp_student_performance.pkl"
)

print("\nMLP model loaded successfully.")


# ------------------------------------------------------------
# 3. Calculate permutation importance
# ------------------------------------------------------------

print("\nCalculating permutation importance...")
print("Please wait...")

importance_result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="accuracy",
    n_repeats=20,
    random_state=42
)

importance_values = importance_result.importances_mean


# ------------------------------------------------------------
# 4. Feature names after preprocessing
# ------------------------------------------------------------

feature_names = [
    "age",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "absences",
    "schoolsup_no",
    "schoolsup_yes",
    "famsup_no",
    "famsup_yes",
    "paid_no",
    "paid_yes",
    "activities_no",
    "activities_yes",
    "higher_no",
    "higher_yes",
    "internet_no",
    "internet_yes"
]


# ------------------------------------------------------------
# 5. Create importance table
# ------------------------------------------------------------

importance_df = pd.DataFrame({
    "Processed_Feature": feature_names,
    "Importance": importance_values
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nProcessed feature importance:")
print("=" * 60)

print(
    importance_df.to_string(
        index=False,
        formatters={
            "Importance": "{:.4f}".format
        }
    )
)


# ------------------------------------------------------------
# 6. Group encoded features into original features
# ------------------------------------------------------------

original_features = {
    "age": ["age"],
    "studytime": ["studytime"],
    "failures": ["failures"],
    "famrel": ["famrel"],
    "freetime": ["freetime"],
    "absences": ["absences"],
    "schoolsup": ["schoolsup_no", "schoolsup_yes"],
    "famsup": ["famsup_no", "famsup_yes"],
    "paid": ["paid_no", "paid_yes"],
    "activities": ["activities_no", "activities_yes"],
    "higher": ["higher_no", "higher_yes"],
    "internet": ["internet_no", "internet_yes"]
}


grouped_importance = []

for original_feature, encoded_features in original_features.items():

    total_importance = importance_df[
        importance_df["Processed_Feature"].isin(encoded_features)
    ]["Importance"].sum()

    grouped_importance.append({
        "Feature": original_feature,
        "Importance": total_importance
    })


grouped_df = pd.DataFrame(
    grouped_importance
)

grouped_df = grouped_df.sort_values(
    by="Importance",
    ascending=False
)


# ------------------------------------------------------------
# 7. Display original feature importance
# ------------------------------------------------------------

print("\nOriginal feature importance:")
print("=" * 60)

print(
    grouped_df.to_string(
        index=False,
        formatters={
            "Importance": "{:.4f}".format
        }
    )
)


# ------------------------------------------------------------
# 8. Create graph
# ------------------------------------------------------------

print("\nCreating feature importance graph...")

plt.figure(figsize=(10, 7))

plt.barh(
    grouped_df["Feature"],
    grouped_df["Importance"]
)

plt.xlabel(
    "Permutation Importance"
)

plt.ylabel(
    "Original Student Feature"
)

plt.title(
    "MLP Model - Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "graphs/feature_importance.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 9. Save results
# ------------------------------------------------------------

grouped_df.to_csv(
    "model/feature_importance.csv",
    index=False
)

print("\nFiles saved successfully:")

print("graphs/feature_importance.png")
print("model/feature_importance.csv")


print("\n" + "=" * 60)
print("STAGE 7.3 COMPLETED!")
print("=" * 60)