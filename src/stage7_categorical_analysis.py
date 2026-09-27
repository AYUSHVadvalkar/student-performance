import pandas as pd
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("STAGE 7.2 - CATEGORICAL FEATURE ANALYSIS")
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

# Categorical features
categorical_features = [
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "higher",
    "internet"
]

# Create graphs folder if it does not exist
os.makedirs("graphs", exist_ok=True)

print("\nPASS percentage for each category:")
print("=" * 60)

results = []

for feature in categorical_features:

    print(f"\nFeature: {feature}")

    # Calculate PASS percentage
    pass_rate = (
        df.groupby(feature)["Result"]
        .apply(lambda x: (x == "PASS").mean() * 100)
    )

    for category, percentage in pass_rate.items():

        print(
            f"  {category}: {percentage:.2f}% PASS"
        )

        results.append({
            "Feature": feature,
            "Category": category,
            "PASS_Percentage": percentage
        })


# Convert results into DataFrame
results_df = pd.DataFrame(results)


# ------------------------------------------------------------
# Create graph
# ------------------------------------------------------------

print("\nCreating categorical feature graph...")

plt.figure(figsize=(12, 7))

labels = []
values = []

for _, row in results_df.iterrows():

    labels.append(
        f"{row['Feature']}\n{row['Category']}"
    )

    values.append(
        row["PASS_Percentage"]
    )

plt.bar(
    range(len(values)),
    values
)

plt.title(
    "PASS Percentage by Categorical Student Features"
)

plt.xlabel(
    "Feature and Category"
)

plt.ylabel(
    "PASS Percentage (%)"
)

plt.xticks(
    range(len(labels)),
    labels,
    rotation=45,
    ha="right"
)

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    "graphs/categorical_feature_analysis.png",
    dpi=300
)

plt.close()


print("\nGraph saved successfully:")
print("graphs/categorical_feature_analysis.png")

print("\n" + "=" * 60)
print("STAGE 7.2 COMPLETED!")
print("=" * 60)