# Insight: The clusters separate clearly, with one cluster showing extreme Monetary and high Frequency (VIP buyers), 
# while another shows very high Recency with low Frequency (inactive customers).
# Business benefit: This confirms segmentation can directly drive retention targeting, 
# VIP loyalty rewards, and personalized campaigns based on customer value.
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Path to the RFM table generated from cleaned retail transactions
rfm_path = "data/rfm_table.csv"

# Load customer-level Recency, Frequency, Monetary values
rfm = pd.read_csv(rfm_path)

# Select the features used for clustering visualization
X = rfm[["Recency", "Frequency", "Monetary"]]

# Standardize features so KMeans clustering remains balanced across RFM values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set number of clusters based on elbow method and business interpretability
k = 4

# Train KMeans model and assign cluster label to each customer
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)

# Create a 3D figure for RFM cluster visualization
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")

# Plot each cluster separately to visualize customer behavior differences
for cluster in sorted(rfm["Cluster"].unique()):
    temp = rfm[rfm["Cluster"] == cluster]
    ax.scatter(
        temp["Recency"],
        temp["Frequency"],
        temp["Monetary"],
        label=f"Cluster {cluster}",
        alpha=0.6
    )

# Add plot title and axis labels for clear interpretation
ax.set_title("3D RFM Clustering Visualization")
ax.set_xlabel("Recency")
ax.set_ylabel("Frequency")
ax.set_zlabel("Monetary")
ax.legend()

# Save the plot for PPT/report submission
plt.tight_layout()
plt.savefig("data/rfm_3d_clusters.png")

# Display the plot window
plt.show()

# Confirmation message in terminal
print("Saved: data/rfm_3d_clusters.png")

# Code Questions

# Q1. Why do we use a 3D plot instead of only 2D scatter plots for clustering?
# Answer: A 3D plot shows how Recency, Frequency, and Monetary interact together in one view.
# It gives stronger proof that clusters are meaningful across all RFM dimensions.
# Q2. Why do we scale the data even though we plot original RFM values in the scatter plot?
# Answer: Scaling is needed for KMeans clustering accuracy, but visualization is clearer in original business units.
# This keeps model correctness and interpretability both strong.

# Plot Interpretation Questions

# Q3. What does it indicate when a cluster has very high Monetary values compared to others?
# Answer: It represents a high-value segment contributing disproportionately to revenue.
# These customers require premium retention offers and loyalty benefits.
# Q4. What does it mean when a cluster has high Recency but low Frequency and Monetary?
# Answer: These customers are inactive and likely churned or close to churn.
# They are the best target for win-back campaigns and limited-time offers.

# Business Use-Case Questions

# Q5. How does this 3D clustering support customer segmentation for targeted marketing campaigns?
# Answer: It provides clear customer groups based on value and engagement behavior.
# Marketing can create separate strategies instead of one generic campaign.
# Q6. How can this plot help inventory and stock optimization strategies?
# Answer: High-frequency clusters reveal which customers drive recurring demand.
# Businesses can prioritize stock for products commonly purchased by loyal segments.