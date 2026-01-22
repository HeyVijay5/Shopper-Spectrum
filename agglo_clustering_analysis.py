import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

rfm_path = "data/rfm_table.csv"

# Load RFM data
rfm = pd.read_csv(rfm_path)

# Select RFM features
X = rfm[["Recency", "Frequency", "Monetary"]]

# Standardize features (important for distance-based clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Running Agglomerative Clustering Model Comparison (Silhouette)...\n")

# ------------------------------------------------------------
# Step 1: Choose best k using Silhouette Score (Agglomerative needs k)
# ------------------------------------------------------------
best_k = None
best_score = -1
scores = []

for k in range(2, 11):
    agg = AgglomerativeClustering(n_clusters=k)
    labels = agg.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    scores.append((k, score))

    print(f"k = {k}, silhouette_score = {score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k

print("\nBest K for Agglomerative based on silhouette score:", best_k)
print("Best silhouette score:", round(best_score, 4))

# Plot Silhouette vs K (like elbow-style selection but for silhouette)
k_vals = [x[0] for x in scores]
sil_vals = [x[1] for x in scores]

plt.figure(figsize=(8, 5))
plt.plot(k_vals, sil_vals, marker="o")
plt.title("Agglomerative Clustering: K vs Silhouette Score")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("data/agglo_k_silhouette.png")
plt.show()

print("\nSaved plot: data/agglo_k_silhouette.png")

# ------------------------------------------------------------
# Step 2: Run final Agglomerative clustering using best k
# ------------------------------------------------------------
final_agg = AgglomerativeClustering(n_clusters=best_k)
rfm["Agglo_Cluster"] = final_agg.fit_predict(X_scaled)

# Plot: Cluster count distribution
cluster_counts = rfm["Agglo_Cluster"].value_counts().sort_index()

plt.figure(figsize=(7, 4))
plt.bar(cluster_counts.index.astype(str), cluster_counts.values)
plt.title("Agglomerative Clustering: Customer Count per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("data/agglo_cluster_distribution.png")
plt.show()

print("Saved plot: data/agglo_cluster_distribution.png")

# Plot: Frequency vs Monetary Scatter
plt.figure(figsize=(7, 5))
for c in sorted(rfm["Agglo_Cluster"].unique()):
    temp = rfm[rfm["Agglo_Cluster"] == c]
    plt.scatter(temp["Frequency"], temp["Monetary"], label=f"Cluster {c}", alpha=0.6)

plt.title("Agglomerative Clustering: Frequency vs Monetary")
plt.xlabel("Frequency")
plt.ylabel("Monetary")
plt.legend()
plt.tight_layout()
plt.savefig("data/agglo_scatter_freq_monetary.png")
plt.show()

print("Saved plot: data/agglo_scatter_freq_monetary.png")

# ------------------------------------------------------------
# Step 3: Save output for verification
# ------------------------------------------------------------
rfm.to_csv("data/customer_segments_agglo.csv", index=False)
print("\nSaved clustered file: data/customer_segments_agglo.csv")


# Q1. Why do we use Silhouette Score for Agglomerative instead of Elbow Method?
# Answer: Elbow is mainly designed for KMeans inertia, not hierarchical clustering.
# Silhouette directly measures cluster separation quality, so it fits Agglomerative better.
# Q2. Why do we still scale the data before Agglomerative clustering?
# Answer: Agglomerative is distance-based, so scaling prevents Monetary from dominating.
# This keeps clustering fair across Recency, Frequency, and Monetary.
# Q3. What does the “Frequency vs Monetary” scatter tell us in Agglomerative results?
# Answer: It shows if customer value segments are clearly separated or heavily overlapping.
# Clear separation means segmentation is actionable for marketing and retention.
# Q4. How do we verify Agglomerative is better than KMeans?
# Answer: We compare silhouette score, cluster balance, and interpretability of segments.
# Even if metrics are close, the final model must produce business-meaningful clusters.

#PLOTS 
# 1) Agglomerative “K vs Silhouette” Plot Insight
# Insight: The best silhouette score occurs at k = 5 (0.6073), meaning hierarchical clustering separates customers slightly better at 5 groups.
# Business benefit: This gives a more detailed segmentation breakdown, useful when marketing wants more granular customer types.

# 2) Customer Count per Cluster (Agglomerative k=5)
# Insight: One cluster is extremely large, and 2 clusters are extremely tiny, meaning the segmentation is highly imbalanced.
# Business benefit: Tiny clusters may represent VIP outliers, but too many micro-clusters can confuse targeting and reduce campaign stability.

# 3) Frequency vs Monetary Scatter Plot (Agglomerative)
# Insight: VIP customers are clearly visible as outliers, but many clusters overlap in the low spend/low frequency region.
# Business benefit: Overlap means marketing actions for low-value customers may not differ much between clusters, so fewer clusters may be more actionable.

# 4) Final Verdict for Agglomerative vs KMeans
# Agglomerative found best k=5, but KMeans still performs better overall for this project because:
# cleaner segmentation + stable cluster sizes + easier mapping into “High Value / Regular / Occasional / At Risk”.