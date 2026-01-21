# Elbow Plot Analysis (What it shows)
# The inertia drops sharply from k=2 to k=4, meaning clustering quality improves a lot in this range. 
# After k=4, the curve starts flattening, so adding more clusters gives smaller gains.
# This indicates the “elbow” is around k=4 (or slightly near k=5), making k=4 a strong and practical choice.
# Why choosing k=4 is best for your project
# k=4 gives a good balance between model performance and business interpretability. 
# It aligns perfectly with your required business segments: High Value, Regular, Occasional, and At Risk.
# Choosing higher k values may over-segment customers and make business actions harder to define.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Path to RFM table created from cleaned transaction data
rfm_path = "data/rfm_table.csv"

# Load the customer-level RFM dataset
rfm = pd.read_csv(rfm_path)

# Select the three key RFM features for clustering
X = rfm[["Recency", "Frequency", "Monetary"]]

# Standardize the features so all columns contribute equally to KMeans distance calculation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Inertia stores the within-cluster sum of squares for each k value
# Lower inertia means customers are closer to their cluster centers
inertia = []

# Try different values of k (number of clusters) from 2 to 10
k_values = range(2, 11)

# Train a KMeans model for each k and store inertia value
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve to visually identify the best k value
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertia, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()

# Save the elbow plot for report / documentation / submission
plt.savefig("data/elbow_plot.png")

# Display the plot window
plt.show()

# Final confirmation message in the terminal
print("Elbow plot saved as data/elbow_plot.png")

# Q1. Why do we scale RFM values using StandardScaler() before applying KMeans?
# Answer: KMeans uses distance-based clustering, so scaling prevents Monetary from dominating due to large values.
# This ensures Recency, Frequency, and Monetary contribute fairly to cluster formation.
# Q2. Why do we compute inertia for multiple k values instead of directly choosing k=4?
# Answer: Trying multiple k values helps measure how cluster compactness improves as we increase clusters.
# The elbow curve provides a data-driven way to select the best k instead of guessing.

# Q3. Why is k=4 a strong business choice for customer segmentation in this project?
# Answer: k=4 naturally aligns with actionable customer groups like High Value, Regular, Occasional, and At Risk.
# It gives meaningful segmentation that marketing and retention teams can apply directly.
# Q4. How does the elbow method support real-world business decisions beyond just model accuracy?
# Answer: It ensures the segmentation is not overcomplicated and remains interpretable for business teams.
# This helps create clear strategies for retention, pricing, and inventory optimization per segment.