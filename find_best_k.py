import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Path to the customer-level RFM dataset created earlier
rfm_path = "data/rfm_table.csv"

# Load RFM dataset for clustering evaluation
rfm = pd.read_csv(rfm_path)

# Select the three key RFM features used for clustering
X = rfm[["Recency", "Frequency", "Monetary"]]

# Standardize features so KMeans distance calculation is not biased
# This ensures Recency, Frequency, and Monetary contribute equally
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Variables to track the best K based on silhouette score
best_k = None
best_score = -1

# Print header for clean output formatting
print("K vs Silhouette Score:\n")

# Try multiple cluster counts from 2 to 10
for k in range(2, 11):
    # Train KMeans for the current k value
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # Get predicted cluster labels for all customers
    labels = kmeans.fit_predict(X_scaled)

    # Compute silhouette score (higher value means better cluster separation)
    score = silhouette_score(X_scaled, labels)
    print(f"k = {k}, silhouette_score = {score:.4f}")

    # Update best_k if current silhouette score is higher than the best score so far
    if score > best_score:
        best_score = score
        best_k = k

# Print final best K selection based on silhouette score
print("\nBest K based on silhouette score:", best_k)
print("Best silhouette score:", round(best_score, 4))

# Q1. Why do we use silhouette score instead of only relying on the elbow method?
# Answer: Silhouette score measures how well-separated clusters are, giving a quantitative evaluation of clustering quality.
# It complements the elbow method, which is mainly a visual heuristic.

# Q2. Why do we test values of k from 2 to 10 specifically?
# Answer: This range is wide enough to observe clustering patterns without creating too many segments.
# It also balances computational cost and interpretability for customer segmentation tasks.

# Q3. Why can the best silhouette score suggest k=2 even though we chose k=4?
# Answer: k=2 often gives the cleanest separation because it forms broad groups like low vs high value customers.
# k=4 is selected for better business interpretability and actionable segmentation.

# Q4. How does selecting a proper k value improve real-time business decisions?
# Answer: The right k gives stable and meaningful customer groups that marketing teams can trust.
# It improves retention targeting, campaign personalization, and revenue optimization strategies.