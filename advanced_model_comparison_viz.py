import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import seaborn as sns

rfm_path = "data/rfm_table.csv"
rfm = pd.read_csv(rfm_path)

X = rfm[["Recency", "Frequency", "Monetary"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# Run all 3 clustering models
# ------------------------------------------------------------

# 1) KMeans
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm["KMeans_Cluster"] = kmeans.fit_predict(X_scaled)

# 2) Agglomerative
agg = AgglomerativeClustering(n_clusters=5)
rfm["Agglo_Cluster"] = agg.fit_predict(X_scaled)

# 3) DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=10)
rfm["DBSCAN_Cluster"] = dbscan.fit_predict(X_scaled)

# ------------------------------------------------------------
# Helper for evaluation metrics
# ------------------------------------------------------------
def compute_metrics(model_name, labels, X_data):
    unique_clusters = set(labels) - {-1}
    clusters_count = len(unique_clusters)

    # If DBSCAN produces <2 clusters, metrics aren't valid
    if clusters_count < 2:
        return {
            "Model": model_name,
            "Silhouette": np.nan,
            "Davies_Bouldin": np.nan,
            "Calinski_Harabasz": np.nan,
            "Clusters": clusters_count
        }

    # Remove noise for DBSCAN metric calculation
    mask = labels != -1
    X_valid = X_data[mask]
    labels_valid = labels[mask]

    return {
        "Model": model_name,
        "Silhouette": silhouette_score(X_valid, labels_valid),
        "Davies_Bouldin": davies_bouldin_score(X_valid, labels_valid),
        "Calinski_Harabasz": calinski_harabasz_score(X_valid, labels_valid),
        "Clusters": clusters_count
    }

metrics = []
metrics.append(compute_metrics("KMeans (k=4)", rfm["KMeans_Cluster"].values, X_scaled))
metrics.append(compute_metrics("Agglomerative (k=5)", rfm["Agglo_Cluster"].values, X_scaled))
metrics.append(compute_metrics("DBSCAN (eps=0.5)", rfm["DBSCAN_Cluster"].values, X_scaled))

metrics_df = pd.DataFrame(metrics)

print("\nModel Comparison Metrics:")
print(metrics_df)

# ============================================================
# PLOT 1: Model Comparison - Silhouette Score
# ============================================================
plt.figure(figsize=(9, 5))
plt.bar(metrics_df["Model"], metrics_df["Silhouette"])
plt.title("Model Comparison: Silhouette Score (Higher is Better)")
plt.xlabel("Model")
plt.ylabel("Silhouette Score")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("data/viz1_silhouette_comparison.png")
plt.show()

# ============================================================
# PLOT 2: Model Comparison - Davies-Bouldin Index
# ============================================================
plt.figure(figsize=(9, 5))
plt.bar(metrics_df["Model"], metrics_df["Davies_Bouldin"])
plt.title("Model Comparison: Davies-Bouldin Index (Lower is Better)")
plt.xlabel("Model")
plt.ylabel("Davies-Bouldin Index")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("data/viz2_davies_bouldin_comparison.png")
plt.show()

# ============================================================
# PLOT 3: Cluster Count Distribution Comparison (Side-by-Side)
# ============================================================
kmeans_counts = rfm["KMeans_Cluster"].value_counts().sort_index()
agg_counts = rfm["Agglo_Cluster"].value_counts().sort_index()

dbscan_counts = rfm[rfm["DBSCAN_Cluster"] != -1]["DBSCAN_Cluster"].value_counts().sort_index()
dbscan_noise = (rfm["DBSCAN_Cluster"] == -1).sum()

plt.figure(figsize=(11, 5))
plt.plot(kmeans_counts.index.astype(str), kmeans_counts.values, marker="o", label="KMeans")
plt.plot(agg_counts.index.astype(str), agg_counts.values, marker="o", label="Agglomerative")
plt.plot(dbscan_counts.index.astype(str), dbscan_counts.values, marker="o", label="DBSCAN (non-noise)")

plt.title("Cluster Size Comparison (Customers per Cluster)")
plt.xlabel("Cluster ID")
plt.ylabel("Number of Customers")
plt.legend()
plt.tight_layout()
plt.savefig("data/viz3_cluster_size_comparison.png")
plt.show()

print("\nDBSCAN Noise Points:", dbscan_noise)

# ============================================================
# PLOT 4: Frequency vs Monetary Scatter (3 Models in 1 Figure)
# ============================================================
plt.figure(figsize=(12, 7))

# KMeans
plt.subplot(1, 3, 1)
plt.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["KMeans_Cluster"], alpha=0.6)
plt.title("KMeans: Frequency vs Monetary")
plt.xlabel("Frequency")
plt.ylabel("Monetary")

# Agglomerative
plt.subplot(1, 3, 2)
plt.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["Agglo_Cluster"], alpha=0.6)
plt.title("Agglomerative: Frequency vs Monetary")
plt.xlabel("Frequency")
plt.ylabel("Monetary")

# DBSCAN (noise will appear as cluster -1)
plt.subplot(1, 3, 3)
plt.scatter(rfm["Frequency"], rfm["Monetary"], c=rfm["DBSCAN_Cluster"], alpha=0.6)
plt.title("DBSCAN: Frequency vs Monetary")
plt.xlabel("Frequency")
plt.ylabel("Monetary")

plt.tight_layout()
plt.savefig("data/viz4_scatter_comparison.png")
plt.show()

# ============================================================
# PLOT 5: Recency vs Monetary Scatter (3 Models in 1 Figure)
# ============================================================
plt.figure(figsize=(12, 7))

# KMeans
plt.subplot(1, 3, 1)
plt.scatter(rfm["Recency"], rfm["Monetary"], c=rfm["KMeans_Cluster"], alpha=0.6)
plt.title("KMeans: Recency vs Monetary")
plt.xlabel("Recency")
plt.ylabel("Monetary")

# Agglomerative
plt.subplot(1, 3, 2)
plt.scatter(rfm["Recency"], rfm["Monetary"], c=rfm["Agglo_Cluster"], alpha=0.6)
plt.title("Agglomerative: Recency vs Monetary")
plt.xlabel("Recency")
plt.ylabel("Monetary")

# DBSCAN
plt.subplot(1, 3, 3)
plt.scatter(rfm["Recency"], rfm["Monetary"], c=rfm["DBSCAN_Cluster"], alpha=0.6)
plt.title("DBSCAN: Recency vs Monetary")
plt.xlabel("Recency")
plt.ylabel("Monetary")

plt.tight_layout()
plt.savefig("data/viz5_scatter_recency_monetary.png")
plt.show()

# ============================================================
# PLOT 6: Cluster Centroids Comparison (KMeans vs Agglomerative)
# Note: DBSCAN doesn't have centroids.
# ============================================================
kmeans_centroids_scaled = kmeans.cluster_centers_
kmeans_centroids = scaler.inverse_transform(kmeans_centroids_scaled)

centroids_df = pd.DataFrame(
    kmeans_centroids,
    columns=["Recency", "Frequency", "Monetary"]
)
centroids_df["Model"] = "KMeans"
centroids_df["Cluster"] = centroids_df.index

plt.figure(figsize=(10, 6))
plt.plot(centroids_df["Cluster"], centroids_df["Recency"], marker="o", label="Recency")
plt.plot(centroids_df["Cluster"], centroids_df["Frequency"], marker="o", label="Frequency")
plt.plot(centroids_df["Cluster"], centroids_df["Monetary"], marker="o", label="Monetary")
plt.title("KMeans Cluster Centroids (Original Scale)")
plt.xlabel("Cluster ID")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig("data/viz6_kmeans_centroids.png")
plt.show()

# ============================================================
# PLOT 7: Heatmap of Cluster Summary Table (MUST BE SECOND LAST)
# ============================================================
kmeans_summary = rfm.groupby("KMeans_Cluster")[["Recency", "Frequency", "Monetary"]].mean()
agg_summary = rfm.groupby("Agglo_Cluster")[["Recency", "Frequency", "Monetary"]].mean()

# DBSCAN excluding noise
dbscan_summary = rfm[rfm["DBSCAN_Cluster"] != -1].groupby("DBSCAN_Cluster")[["Recency", "Frequency", "Monetary"]].mean()

combined_summary = pd.concat(
    [
        kmeans_summary.rename_axis("Cluster").reset_index().assign(Model="KMeans"),
        agg_summary.rename_axis("Cluster").reset_index().assign(Model="Agglomerative"),
        dbscan_summary.rename_axis("Cluster").reset_index().assign(Model="DBSCAN"),
    ],
    ignore_index=True
)

heatmap_data = combined_summary.pivot_table(
    index=["Model", "Cluster"],
    values=["Recency", "Frequency", "Monetary"]
)

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=False, cmap="Blues")
plt.title("Cluster Behavior Heatmap (RFM Means)")
plt.tight_layout()
plt.savefig("data/viz7_cluster_heatmap.png")
plt.show()

# ============================================================
# CORRELATION PLOT (MUST BE LAST)
# ============================================================
corr = rfm[["Recency", "Frequency", "Monetary"]].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix: Recency vs Frequency vs Monetary")
plt.tight_layout()
plt.savefig("data/viz8_correlation_matrix.png")
plt.show()

print("\nSaved all visualizations into data/ folder.")
