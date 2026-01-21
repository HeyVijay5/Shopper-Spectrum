import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Path to the RFM table created from cleaned retail transaction data
rfm_path = "data/rfm_table.csv"

# Load the RFM dataset for training the clustering model
rfm = pd.read_csv(rfm_path)

# Select RFM features used for KMeans clustering
X = rfm[["Recency", "Frequency", "Monetary"]]

# Standardize RFM values so clustering is not biased toward large Monetary values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set number of clusters based on elbow method and business interpretability
k = 4

# Train the KMeans model on scaled data
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Save scaler for consistent preprocessing during Streamlit prediction
joblib.dump(scaler, "models/scaler.pkl")

# Save trained KMeans model for real-time cluster prediction in the Streamlit app
joblib.dump(kmeans, "models/kmeans_model.pkl")

# Print confirmation messages to verify the model files are saved successfully
print("Saved models:")
print("models/scaler.pkl")
print("models/kmeans_model.pkl")

# Q1. Why do we save the scaler and KMeans model separately using Joblib?
# Answer: Streamlit needs the same scaler to preprocess user inputs before applying the saved clustering model.
# Saving both ensures prediction consistency between training and real-time usage.

# Q2. Why do we train the model on scaled data instead of raw RFM values?
# Answer: Scaling prevents Monetary values from dominating distance computations in KMeans clustering.
# This produces more balanced and meaningful customer segments.

# Q3. How does saving the KMeans model support real-time business use cases?
# Answer: It enables instant customer segmentation for marketing, retention, and personalization workflows.
# Real-time predictions help businesses react faster to changing customer behavior.

# Q4. Why is KMeans a practical choice for customer segmentation in this project?
# Answer: It is simple, fast, and works well for grouping customers based on behavioral similarity.
# It produces interpretable segments that can be mapped to actionable business strategies.