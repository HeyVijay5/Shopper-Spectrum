import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

rfm_path = "data/rfm_table.csv"

# Load RFM dataset
rfm = pd.read_csv(rfm_path)

# Select RFM features
X = rfm[["Recency", "Frequency", "Monetary"]]

# Standardize because DBSCAN is distance-based
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Running DBSCAN parameter testing...\n")

# ------------------------------------------------------------
# Step 1: Try different eps values to find meaningful clusters
# DBSCAN doesn't use k, so eps acts like the clustering sensitivity control
# ------------------------------------------------------------
eps_values = [0.3, 0.5, 0.7, 0.8, 1.0, 1.2]
min_samples = 10

best_eps = None
best_score = -1
best_labels = None

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    # Ignore noise label (-1)
    unique_clusters = set(labels) - {-1}

    # Silhouette requires at least 2 clusters (excluding noise)
    if len(unique_clusters) < 2:
        print(f"eps={eps} -> clusters={len(unique_clusters)} (Not valid)")
        continue

    # Evaluate silhouette only on non-noise points
    mask = labels != -1
    score = silhouette_score(X_scaled[mask], labels[mask])

    print(f"eps={eps} -> clusters={len(unique_clusters)}, silhouette={score:.4f}")

    if score > best_score:
        best_score = score
        best_eps = eps
        best_labels = labels

print("\nBest DBSCAN eps:", best_eps)
print("Best silhouette score:", round(best_score, 4))

# ------------------------------------------------------------
# Step 2: Run final DBSCAN using best eps
# ------------------------------------------------------------
final_dbscan = DBSCAN(eps=best_eps, min_samples=min_samples)
rfm["DBSCAN_Cluster"] = final_dbscan.fit_predict(X_scaled)

# Count clusters and noise points
cluster_counts = rfm["DBSCAN_Cluster"].value_counts()

print("\nDBSCAN Cluster Counts (including noise = -1):")
print(cluster_counts)

# Save output
rfm.to_csv("data/customer_segments_dbscan.csv", index=False)
print("\nSaved clustered file: data/customer_segments_dbscan.csv")

# ------------------------------------------------------------
# Plot 1: Cluster distribution (excluding noise for clarity)
# ------------------------------------------------------------
valid_clusters = rfm[rfm["DBSCAN_Cluster"] != -1]["DBSCAN_Cluster"].value_counts().sort_index()

plt.figure(figsize=(7, 4))
plt.bar(valid_clusters.index.astype(str), valid_clusters.values)
plt.title("DBSCAN: Customer Count per Cluster (Noise excluded)")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("data/dbscan_cluster_distribution.png")
plt.show()

print("Saved plot: data/dbscan_cluster_distribution.png")

# ------------------------------------------------------------
# Plot 2: Frequency vs Monetary scatter with noise points
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))

# Noise points (-1) in gray
noise = rfm[rfm["DBSCAN_Cluster"] == -1]
plt.scatter(noise["Frequency"], noise["Monetary"], label="Noise (-1)", alpha=0.4)

# Actual clusters
for c in sorted(set(rfm["DBSCAN_Cluster"].unique()) - {-1}):
    temp = rfm[rfm["DBSCAN_Cluster"] == c]
    plt.scatter(temp["Frequency"], temp["Monetary"], label=f"Cluster {c}", alpha=0.6)

plt.title("DBSCAN: Frequency vs Monetary (with Noise)")
plt.xlabel("Frequency")
plt.ylabel("Monetary")
plt.legend()
plt.tight_layout()
plt.savefig("data/dbscan_scatter_freq_monetary.png")
plt.show()

print("Saved plot: data/dbscan_scatter_freq_monetary.png")

# Q1. Why can’t we use Elbow Method for DBSCAN?
# Answer: DBSCAN does not use k clusters, so inertia-based elbow is not applicable.
# Instead, tuning eps and min_samples controls clustering behavior.
# Q2. Why do DBSCAN results often show “noise points (-1)” ?
# Answer: DBSCAN identifies outliers that don’t belong to any dense cluster.
# This is useful for fraud/outlier detection but may reduce segmentation coverage.
# Q3. Why do we evaluate silhouette after removing noise points?
# Answer: Noise points are not part of any cluster and distort cluster separation metrics.
# Evaluating only valid points gives a more realistic quality score.
# Q4. When is DBSCAN better than KMeans for business segmentation?
# Answer: DBSCAN is better when clusters are irregular shapes and outliers matter more than coverage.
# For marketing segmentation, KMeans is usually better because it gives stable, complete grouping.