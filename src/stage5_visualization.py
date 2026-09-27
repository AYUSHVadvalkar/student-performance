import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("STAGE 5 - MODEL VISUALIZATION")
print("=" * 60)

# ============================================================
# 5.1 PASS vs FAIL DISTRIBUTION
# ============================================================

print("\nCreating PASS vs FAIL distribution...")

df = pd.read_csv("dataset/student-mat.csv", sep=";")

df["Result"] = df["G3"].apply(
    lambda x: "PASS" if x >= 10 else "FAIL"
)

result_counts = df["Result"].value_counts()

print("\nStudent Result Distribution:")
print(result_counts)

plt.figure(figsize=(7, 5))

plt.bar(
    result_counts.index,
    result_counts.values
)

plt.title("Student Performance Distribution")
plt.xlabel("Result")
plt.ylabel("Number of Students")

for i, value in enumerate(result_counts.values):
    plt.text(
        i,
        value + 3,
        str(value),
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "graphs/pass_fail_distribution.png",
    dpi=300
)

plt.close()

print("PASS/FAIL graph saved successfully.")


# ============================================================
# 5.2 MODEL PERFORMANCE METRICS
# ============================================================

print("\nCreating model performance graph...")

metrics = {
    "Accuracy": 67.09,
    "Precision": 68.00,
    "Recall": 96.23,
    "F1-Score": 79.69
}

print("\nModel Performance:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.2f}%")

plt.figure(figsize=(8, 5))

names = list(metrics.keys())
values = list(metrics.values())

plt.bar(names, values)

plt.title("MLP Model Performance Metrics")
plt.xlabel("Evaluation Metric")
plt.ylabel("Percentage (%)")

plt.ylim(0, 100)

for i, value in enumerate(values):
    plt.text(
        i,
        value + 2,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "graphs/model_performance_metrics.png",
    dpi=300
)

plt.close()

print("\nModel performance graph saved successfully:")
print("graphs/model_performance_metrics.png")


print("\n" + "=" * 60)
print("STAGE 5.2 COMPLETED!")
print("=" * 60)