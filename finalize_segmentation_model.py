import pandas as pd
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Path to the customer-level RFM table generated from cleaned transaction data
rfm_path = "data/rfm_table.csv"

# Load the RFM dataset containing Recency, Frequency, and Monetary for each customer
rfm = pd.read_csv(rfm_path)

# Select only RFM features for clustering
X = rfm[["Recency", "Frequency", "Monetary"]]

# Scale the RFM features so KMeans distance calculations are balanced
# This prevents Monetary from dominating due to larger numeric values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Final number of clusters selected based on elbow method and business interpretability
k = 4

# Train the final KMeans model and assign clusters to each customer
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)

# Final confirmed mapping from cluster ID to business-friendly segment label
# This converts technical clusters into actionable marketing segments
segment_map = {
    0: "Occasional",
    1: "At Risk",
    2: "High Value",
    3: "Regular"
}

# Save the scaler so Streamlit can scale user inputs in the same way as training
joblib.dump(scaler, "models/scaler.pkl")

# Save the trained KMeans model so Streamlit can predict customer clusters instantly
joblib.dump(kmeans, "models/kmeans_model_k4.pkl")

# Save the segment label mapping so Streamlit can display meaningful segment names
with open("models/segment_map.json", "w") as f:
    json.dump(segment_map, f)

# Print confirmations so we can verify all required deployment files are saved
print("Saved:")
print("models/scaler.pkl")
print("models/kmeans_model_k4.pkl")
print("models/segment_map.json")


# Q1. Why do we save scaler.pkl and not just the KMeans model?
# Answer: Streamlit inputs must be scaled exactly like training for the KMeans predictions to remain accurate.
# Saving the scaler ensures consistent preprocessing and avoids prediction mismatch.

# Q2. Why do we save segment_map.json instead of hardcoding segment names in Streamlit?
# Answer: It keeps labeling logic separate from UI code and makes it easier to update segments later.
# This improves maintainability and ensures all modules use one consistent mapping.

# Q3. Why is it important to map clusters to segments like “High Value” and “At Risk”?
# Answer: Business teams need interpretable labels to take immediate actions like retention and upselling.
# Numeric clusters alone do not explain customer behavior or decision-making.

# Q4. How does finalizing and saving this model support real-time business use cases?
# Answer: It enables instant customer segmentation during campaigns, checkout flows, or CRM analysis.
# This improves targeted marketing, churn prevention, and personalized engagement at scale.