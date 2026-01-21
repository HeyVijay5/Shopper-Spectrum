# Plot 1: Customer Count per Cluster
# Insight: Cluster 0 contains the majority of customers, while Cluster 2 is extremely small, indicating a rare VIP-like segment.
# Business benefit: Helps allocate marketing budgets efficiently—mass campaigns for Cluster 0 and premium retention for Cluster 2.

# Plot 2: Frequency vs Monetary Scatter Plot (Cluster Visualization)
# Insight: Cluster 2 clearly stands out with very high monetary and/or high frequency customers, while Cluster 0 and 1 mostly remain low-value.
# Business benefit: Confirms real customer value separation, enabling targeted upsell for Cluster 3 and churn control for Cluster 1.


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Path to the customer-level RFM dataset created earlier
rfm_path = "data/rfm_table.csv"

# Load RFM data containing Recency, Frequency, and Monetary for each customer
rfm = pd.read_csv(rfm_path)

# Select RFM features for clustering
X = rfm[["Recency", "Frequency", "Monetary"]]

# Scale features to ensure equal contribution in distance-based clustering (KMeans)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set number of clusters based on elbow method and business interpretability
k = 4

# Train KMeans and assign a cluster label to each customer
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)

# ------------------------------------------------------------
# Create cluster summary for business interpretation
# This helps understand what each cluster represents using average RFM values
# ------------------------------------------------------------
cluster_summary = rfm.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean"),
    Median_Monetary=("Monetary", "median")
).reset_index()

# Print cluster summary for verification and report documentation
print("Cluster Summary:")
print(cluster_summary)

# ------------------------------------------------------------
# Save outputs to reuse in segmentation labeling and Streamlit dashboard
# ------------------------------------------------------------
rfm.to_csv("data/customer_segments.csv", index=False)
cluster_summary.to_csv("data/cluster_summary.csv", index=False)

print("\nSaved:")
print("data/customer_segments.csv")
print("data/cluster_summary.csv")

# ------------------------------------------------------------
# Plot 1: Cluster size distribution (customers per cluster)
# Helps identify which segment represents most of the customer base
# ------------------------------------------------------------
plt.figure(figsize=(7, 4))
plt.bar(cluster_summary["Cluster"].astype(str), cluster_summary["Customers"])
plt.title("Customer Count per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("data/cluster_distribution.png")
plt.show()

print("Saved plot: data/cluster_distribution.png")

# ------------------------------------------------------------
# Plot 2: Scatter plot visualization (Frequency vs Monetary)
# Helps visualize separation between clusters and identify high-value groups
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))
for c in sorted(rfm["Cluster"].unique()):
    temp = rfm[rfm["Cluster"] == c]
    plt.scatter(temp["Frequency"], temp["Monetary"], label=f"Cluster {c}", alpha=0.6)

plt.title("Cluster Visualization: Frequency vs Monetary")
plt.xlabel("Frequency")
plt.ylabel("Monetary")
plt.legend()
plt.tight_layout()
plt.savefig("data/cluster_scatter_freq_monetary.png")
plt.show()

print("Saved plot: data/cluster_scatter_freq_monetary.png")

# Code Questions

# Q1. Why do we include both Average Monetary and Median Monetary in the cluster summary?
# Answer: Average can be distorted by extreme spenders, while median shows typical customer behavior in the cluster.
# Using both improves interpretation and makes business decisions more reliable.
# Q2. Why do we visualize clusters using Frequency vs Monetary instead of plotting all three RFM features?
# Answer: Frequency and Monetary directly represent customer purchasing strength and business value.
# This plot gives a clear segmentation story without overcomplicating the visualization.
# Q3. Why do we save plots like cluster_distribution.png and cluster_scatter_freq_monetary.png?
# Answer: Saved plots are essential for project submission, PPT reporting, and reproducibility.
# They also make your analysis traceable and easy to review.
# Q4. Why do we set random_state=42 and n_init=10 in KMeans?
# Answer: random_state ensures consistent results across runs for stable segmentation.
# n_init=10 improves clustering reliability by trying multiple initial centroid starts.

# Plot Interpretation Questions

# Q5. What does it mean when one cluster is very large and another is very small?
# Answer: A large cluster usually represents the majority customer behavior (mass segment).
# A small cluster often represents VIPs or rare patterns with high business importance.
# Q6. Why do some points appear as extreme outliers in Monetary values?
# Answer: These are customers placing bulk orders or buying expensive combinations frequently.
# They usually represent high-value accounts worth special retention treatment.
# Q7. What does it indicate if Cluster 1 appears low in Frequency and low in Monetary?
# Answer: This cluster typically represents low engagement customers with low business value.
# It becomes the best target for reactivation campaigns and personalized discounts.
# Q8. Why is the scatter plot useful even if the clusters overlap in the lower region?
# Answer: Overlap is expected because many customers behave similarly at low frequency/spend.
# The plot still highlights separation in high-value zones, which matters most for business decisions.

# Business Use-Case Questions

# Q9. How does cluster distribution support targeted marketing campaigns?
# Answer: It helps decide which segments need mass promotions versus premium personalization.
# This improves ROI by matching spend strategy to customer value.
# Q10. How does Frequency vs Monetary clustering support product recommendation strategies?
# Answer: High frequency segments respond well to cross-selling and personalized recommendations.
# Low frequency segments need discovery-based suggestions to increase engagement.
# Q11. How does this help identify at-risk customers for retention programs?
# Answer: Customers with low frequency and high recency trends are likely churn candidates.
# Retention actions can be triggered early to recover revenue.
# Q12. How does this clustering help in inventory optimization?
# Answer: High-value clusters often purchase specific high-demand products repeatedly.
# Stocking those products properly prevents stockouts and supports stable revenue flow.