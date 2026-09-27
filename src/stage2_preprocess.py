import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/student-mat.csv", sep=";")


# ============================================================
# 2. SELECT FEATURES
# ============================================================

features = [
    "age",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "higher",
    "internet",
    "famrel",
    "freetime",
    "absences"
]


X = df[features].copy()


# ============================================================
# 3. CREATE TARGET
# ============================================================

# PASS = 1
# FAIL = 0

y = df["G3"].apply(
    lambda x: 1 if x >= 10 else 0
)


# ============================================================
# 4. DEFINE NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = [
    "age",
    "studytime",
    "failures",
    "famrel",
    "freetime",
    "absences"
]

categorical_features = [
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "higher",
    "internet"
]


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. CREATE PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 7. FIT ONLY ON TRAINING DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# ============================================================
# 8. CONVERT TO NUMPY ARRAYS
# ============================================================

X_train_processed = np.asarray(X_train_processed)
X_test_processed = np.asarray(X_test_processed)

y_train = np.asarray(y_train)
y_test = np.asarray(y_test)


# ============================================================
# 9. SAVE PREPROCESSOR
# ============================================================

joblib.dump(
    preprocessor,
    "model/preprocessor.pkl"
)


# ============================================================
# 10. SAVE PROCESSED DATA
# ============================================================

np.save(
    "model/X_train.npy",
    X_train_processed
)

np.save(
    "model/X_test.npy",
    X_test_processed
)

np.save(
    "model/y_train.npy",
    y_train
)

np.save(
    "model/y_test.npy",
    y_test
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("=" * 60)
print("STAGE 2 - DATA PREPROCESSING")
print("=" * 60)

print("\nOriginal dataset shape:")
print(df.shape)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)

print("\nProcessed training data shape:")
print(X_train_processed.shape)

print("\nProcessed testing data shape:")
print(X_test_processed.shape)

print("\nTraining PASS/FAIL distribution:")
print(pd.Series(y_train).value_counts())

print("\nTesting PASS/FAIL distribution:")
print(pd.Series(y_test).value_counts())

print("\nPreprocessing completed successfully!")

print("\nFiles saved:")
print("model/preprocessor.pkl")
print("model/X_train.npy")
print("model/X_test.npy")
print("model/y_train.npy")
print("model/y_test.npy")

print("\nSTAGE 2 COMPLETED!")