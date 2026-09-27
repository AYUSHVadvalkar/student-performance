import matplotlib.pyplot as plt

print("=" * 60)
print("STAGE 6.3 - BASELINE vs BALANCED MODEL COMPARISON")
print("=" * 60)

# Results from Stage 4
baseline = {
    "Accuracy": 67.09,
    "Precision": 68.00,
    "PASS Recall": 96.23,
    "PASS F1": 79.69
}

# Results from Stage 6.2
balanced = {
    "Accuracy": 69.62,
    "Precision": 74.58,
    "PASS Recall": 83.02,
    "PASS F1": 78.57
}

print("\nBaseline MLP:")
for metric, value in baseline.items():
    print(f"{metric}: {value:.2f}%")

print("\nBalanced MLP:")
for metric, value in balanced.items():
    print(f"{metric}: {value:.2f}%")

# Create comparison graph
metrics = list(baseline.keys())

baseline_values = list(baseline.values())
balanced_values = list(balanced.values())

x = range(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    baseline_values,
    width,
    label="Baseline MLP"
)

plt.bar(
    [i + width / 2 for i in x],
    balanced_values,
    width,
    label="Balanced MLP"
)

plt.title("Baseline vs Balanced MLP Model")
plt.xlabel("Performance Metric")
plt.ylabel("Percentage (%)")

plt.xticks(list(x), metrics)

plt.ylim(0, 100)

plt.legend()

# Add values above bars
for i, value in enumerate(baseline_values):
    plt.text(
        i - width / 2,
        value + 2,
        f"{value:.1f}%",
        ha="center"
    )

for i, value in enumerate(balanced_values):
    plt.text(
        i + width / 2,
        value + 2,
        f"{value:.1f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "graphs/baseline_vs_balanced.png",
    dpi=300
)

plt.close()

print("\nComparison graph saved successfully:")
print("graphs/baseline_vs_balanced.png")

print("\n" + "=" * 60)
print("STAGE 6.3 COMPLETED!")
print("=" * 60)