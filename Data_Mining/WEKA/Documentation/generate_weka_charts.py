import os
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.join(
    "Data_Mining", "WEKA", "Documentation", "Charts"
)

CLASS_DIR = os.path.join(BASE, "Classification")
CLUSTER_DIR = os.path.join(BASE, "Clustering")

os.makedirs(CLASS_DIR, exist_ok=True)
os.makedirs(CLUSTER_DIR, exist_ok=True)

# ==========================================
# CLASSIFICATION RESULTS
# ==========================================

algorithms = ["J48", "Random Forest", "Naive Bayes", "SMO"]

accuracy = [53.0106, 52.4905, 53.8708, 55.211]
precision = [0.475, 0.492, 0.478, 0.428]
recall = [0.530, 0.525, 0.539, 0.552]
f_measure = [0.483, 0.501, 0.466, 0.413]
kappa = [0.1029, 0.1335, 0.0971, 0.0213]
mae = [0.3709, 0.3642, 0.3799, 0.3592]
rmse = [0.4725, 0.4573, 0.4440, 0.4594]

def save_bar_chart(labels, values, title, ylabel, filename,
                   output_dir, percentage=False):

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values)

    plt.title(title, fontsize=15, fontweight="bold")
    plt.xlabel("Algorithms")
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    if percentage:
        plt.ylim(0, max(values) * 1.2)

    for bar, value in zip(bars, values):
        label = f"{value:.2f}%" if percentage else f"{value:.4f}"
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# Six classification metric charts

save_bar_chart(
    algorithms, accuracy,
    "Classification Accuracy Comparison",
    "Accuracy (%)", "01_accuracy.png",
    CLASS_DIR, True
)

save_bar_chart(
    algorithms, precision,
    "Classification Precision Comparison",
    "Precision", "02_precision.png", CLASS_DIR
)

save_bar_chart(
    algorithms, recall,
    "Classification Recall Comparison",
    "Recall", "03_recall.png", CLASS_DIR
)

save_bar_chart(
    algorithms, f_measure,
    "Classification F-Measure Comparison",
    "F-Measure", "04_f_measure.png", CLASS_DIR
)

save_bar_chart(
    algorithms, kappa,
    "Classification Kappa Comparison",
    "Kappa Statistic", "05_kappa.png", CLASS_DIR
)

save_bar_chart(
    algorithms, rmse,
    "Classification RMSE Comparison",
    "RMSE", "06_rmse.png", CLASS_DIR
)

# Combined precision, recall and F-measure chart

x = np.arange(len(algorithms))
width = 0.24

plt.figure(figsize=(11, 6))
plt.bar(x - width, precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar(x + width, f_measure, width, label="F-measure")

plt.title("Classification Performance Metrics",
          fontsize=15, fontweight="bold")
plt.xlabel("Algorithms")
plt.ylabel("Score")
plt.xticks(x, algorithms)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(
    os.path.join(CLASS_DIR, "07_combined_metrics.png"),
    dpi=300
)
plt.close()


# ==========================================
# CLUSTERING RESULTS
# ==========================================

cluster_algorithms = ["SimpleKMeans", "EM"]

correct = [42.0084, 44.6489]
incorrect = [57.9916, 55.3511]
build_time = [0.14, 0.51]

# Actual cluster distributions

kmeans_distribution = [2819, 962, 1218]
em_distribution = [3332, 1006, 661]

cluster_labels = ["Cluster 0", "Cluster 1", "Cluster 2"]

# Chart 1: Clustering agreement

x = np.arange(2)
width = 0.35

plt.figure(figsize=(9, 6))
plt.bar(x - width/2, correct, width, label="Correctly clustered")
plt.bar(x + width/2, incorrect, width, label="Incorrectly clustered")

plt.title("Clustering Agreement Comparison",
          fontsize=15, fontweight="bold")
plt.xlabel("Clustering Algorithms")
plt.ylabel("Instances (%)")
plt.xticks(x, cluster_algorithms)
plt.ylim(0, 70)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig(
    os.path.join(CLUSTER_DIR, "01_clustering_agreement.png"),
    dpi=300
)
plt.close()

# Chart 2: SimpleKMeans distribution

plt.figure(figsize=(9, 6))
bars = plt.bar(cluster_labels, kmeans_distribution)

plt.title("SimpleKMeans Cluster Distribution",
          fontsize=15, fontweight="bold")
plt.xlabel("Clusters")
plt.ylabel("Number of Instances")
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar, value in zip(bars, kmeans_distribution):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        str(value),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    os.path.join(CLUSTER_DIR, "02_simplekmeans_distribution.png"),
    dpi=300
)
plt.close()

# Chart 3: EM distribution

plt.figure(figsize=(9, 6))
bars = plt.bar(cluster_labels, em_distribution)

plt.title("EM Cluster Distribution",
          fontsize=15, fontweight="bold")
plt.xlabel("Clusters")
plt.ylabel("Number of Instances")
plt.grid(axis="y", linestyle="--", alpha=0.4)

for bar, value in zip(bars, em_distribution):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        str(value),
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(
    os.path.join(CLUSTER_DIR, "03_em_distribution.png"),
    dpi=300
)
plt.close()

# Chart 4: Model building time

save_bar_chart(
    cluster_algorithms,
    build_time,
    "Clustering Model Building Time",
    "Time (seconds)",
    "04_build_time.png",
    CLUSTER_DIR
)

# Summary

print("\n======================================")
print("CINESCOPE WEKA CHARTS GENERATED")
print("======================================")

print("\nCLASSIFICATION CHARTS:")
for file in sorted(os.listdir(CLASS_DIR)):
    print(" -", file)

print("\nCLUSTERING CHARTS:")
for file in sorted(os.listdir(CLUSTER_DIR)):
    print(" -", file)

print("\nClassification charts:", len(os.listdir(CLASS_DIR)))
print("Clustering charts:", len(os.listdir(CLUSTER_DIR)))
print("\nAll charts saved successfully!")
