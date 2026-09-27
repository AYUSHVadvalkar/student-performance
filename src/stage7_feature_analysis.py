import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("STAGE 7.1 - FEATURE ANALYSIS")
print("=" * 60)

# Load dataset
df = pd.read_csv(
    "dataset/student-mat.csv",
    sep=";"
)

# Create PASS / FAIL result
df["Result"] = df["G3"].apply(
    lambda x: "PASS" if x >= 10 else "FAIL"
)

# Features to analyze
features = [
    "age",
    "studytime",
    "failures",
    "absences",
    "famrel",
    "freetime"
]

print("\nAverage feature values by student result:")
print("=" * 60)

# Calculate averages
average_values = df.groupby("Result")[features].mean()

print(average_values.round(2))


# ------------------------------------------------------------
# Create comparison graph
# ------------------------------------------------------------

print("\nCreating feature comparison graph...")

# Select features that are easier to compare
plot_features = [
    "studytime",
    "failures",
    "absences",
    "famrel",
    "freetime"
]

pass_values = average_values.loc["PASS", plot_features]
fail_values = average_values.loc["FAIL", plot_features]

x = range(len(plot_features))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    fail_values,
    width,
    label="FAIL"
)

plt.bar(
    [i + width / 2 for i in x],
    pass_values,
    width,
    label="PASS"
)

plt.title("Average Student Features: PASS vs FAIL")
plt.xlabel("Student Feature")
plt.ylabel("Average Value")

plt.xticks(
    list(x),
    [
        "Study Time",
        "Failures",
        "Absences",
        "Family Relationship",
        "Free Time"
    ]
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "graphs/feature_comparison.png",
    dpi=300
)

plt.close()

print("\nFeature comparison graph saved successfully:")
print("graphs/feature_comparison.png")


print("\n" + "=" * 60)
print("STAGE 7.1 COMPLETED!")
print("=" * 60)