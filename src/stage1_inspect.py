import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("dataset/student-mat.csv", sep=";")
print("=" * 60)
print("STUDENT PERFORMANCE DATASET")
print("=" * 60)

# ==========================================
# 2. FIRST FIVE ROWS
# ==========================================

print("\nFIRST 5 ROWS:")
print(df.head())

# ==========================================
# 3. DATASET SIZE
# ==========================================

print("\nDATASET SHAPE:")
print(df.shape)

# ==========================================
# 4. COLUMN NAMES
# ==========================================

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# ==========================================
# 5. DATA INFORMATION
# ==========================================

print("\nDATASET INFORMATION:")
df.info()

# ==========================================
# 6. MISSING VALUES
# ==========================================

print("\nMISSING VALUES:")
print(df.isnull().sum())

# ==========================================
# 7. CREATE PASS / FAIL TARGET
# ==========================================

df["result"] = df["G3"].apply(
    lambda x: "PASS" if x >= 10 else "FAIL"
)

print("\nPASS / FAIL DISTRIBUTION:")
print(df["result"].value_counts())

# ==========================================
# 8. SELECT FEATURES
# ==========================================

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

X = df[features]
y = df["result"]

# ==========================================
# 9. DISPLAY INPUT AND TARGET
# ==========================================

print("\nSELECTED INPUT FEATURES:")
print(X.head())

print("\nTARGET VALUES:")
print(y.head())

print("\nNUMBER OF INPUT FEATURES:", len(features))

print("\nSTAGE 1 COMPLETED!")