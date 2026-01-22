# Cluster Distribution Plot Analysis (Key Insights)

# Most customers belong to Cluster 0, meaning your business has a large base of low-to-mid engagement buyers.
# Cluster 1 is the second-largest group and typically represents low frequency / high recency customers (potential churn risk). 
# Cluster 2 is extremely small, which usually indicates a VIP segment with very high spending or bulk buying behavior.
# This distribution confirms the classic retail pattern: a majority of customers are occasional buyers, 
# while a very small segment contributes disproportionately to revenue. 
# This makes segmentation highly valuable because marketing actions should differ between mass customers and VIP customers.
# It also highlights where retention programs can create the highest ROI, especially for customers close to churn.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Path to RFM dataset generated from cleaned retail transactions
rfm_path = "data/rfm_table.csv"

# Load the RFM table containing customer-level Recency, Frequency, and Monetary values
rfm = pd.read_csv(rfm_path)

# Selecting features used for clustering
X = rfm[["Recency", "Frequency", "Monetary"]]

# Standardizing RFM features so KMeans is not biased toward larger values (especially Monetary)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Final selected number of clusters based on elbow method and business interpretability
k = 4

# Train KMeans clustering model and assign each customer to a cluster
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)

# ------------------------------------------------------------
# Create a cluster summary table to understand each cluster profile
# This helps convert technical clusters into meaningful business segments
# ------------------------------------------------------------
cluster_summary = rfm.groupby("Cluster").agg(
    Customers=("CustomerID", "count"),
    Avg_Recency=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary=("Monetary", "mean")
).reset_index()

# Print summary for quick verification and report usage
print("Cluster Summary:")
print(cluster_summary)

# ------------------------------------------------------------
# Save customer cluster assignments and cluster summary for later use
# These files are required for Streamlit integration and business analysis
# ------------------------------------------------------------
rfm.to_csv("data/customer_segments.csv", index=False)
cluster_summary.to_csv("data/cluster_summary.csv", index=False)

print("\nSaved:")
print("data/customer_segments.csv")
print("data/cluster_summary.csv")

# ------------------------------------------------------------
# Visualize cluster distribution (how customers are spread across clusters)
# This highlights whether customers are mostly low-value, regular, or VIP groups
# ------------------------------------------------------------
plt.figure(figsize=(7, 4))
cluster_summary_sorted = cluster_summary.sort_values("Customers", ascending=False)
plt.bar(cluster_summary_sorted["Cluster"].astype(str), cluster_summary_sorted["Customers"])
plt.title("Customer Count per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()

# Save plot for PPT/report and analysis documentation
plt.savefig("data/cluster_distribution.png")

# Display the plot window
plt.show()

# Print confirmation that plot is saved
print("Saved plot: data/cluster_distribution.png")


# Q1. Why do we standardize RFM features before applying KMeans clustering?
# Answer: KMeans depends on distance calculations, so scaling prevents Monetary from overpowering other features.
# This ensures clustering reflects balanced customer behavior across Recency, Frequency, and Monetary.
# Q2. Why do we generate cluster_summary after clustering?
# Answer: Cluster summary helps interpret each cluster by analyzing average Recency, Frequency, and Monetary values.
# It converts raw cluster numbers into meaningful customer segments for business decisions.

# Q3. What does the large size of Cluster 0 indicate from a business perspective?
# Answer: It suggests most customers are occasional or moderate-value buyers who contribute smaller individual revenue.
# This segment is the best target for personalized offers and recommendation-driven upselling.
# Q4. Why is it important that one cluster is very small compared to others?
# Answer: A small cluster often represents VIP customers or rare purchasing patterns with very high spending behavior.
# Protecting this segment through premium loyalty and retention strategies can secure major revenue impact.