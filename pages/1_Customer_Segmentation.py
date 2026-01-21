# Q1. Why do we load scaler.pkl and kmeans_model_k4.pkl instead of training inside Streamlit?
# Answer: Loading saved models ensures predictions are fast and consistent in real time.
# It also avoids retraining every refresh, which improves performance and reliability.
# Q2. Why do we scale the input using scaler.transform() before predicting the cluster?
# Answer: Scaling keeps Recency, Frequency, and Monetary on the same numeric range.
# Without scaling, Monetary dominates the model and clustering becomes biased.

# -KEY BUSINESS INSIGHTS 

# Q3. Why did we use RFM-based KMeans segmentation for targeted marketing campaigns?
# Answer: RFM captures customer value and activity patterns in a simple, business-readable way.
# KMeans groups customers into actionable segments like High Value and At Risk for campaigns.
# Q4. How does this segmentation help identify at-risk customers and improve retention?
# Answer: At-risk customers show high recency and low frequency, meaning they stopped purchasing.
# This lets businesses run win-back offers early and reduce churn efficiently.

import streamlit as st
import numpy as np
import joblib
import json

# Page configuration for better layout and page title
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Page heading and short description
st.title("Customer Segmentation")
st.write("Predict customer segment using RFM values (Recency, Frequency, Monetary).")

# ------------------------------------------------------------
# Loading the saved ML models (trained earlier)
# This avoids retraining the model every time the app runs
# ------------------------------------------------------------
scaler = joblib.load("models/scaler.pkl")
kmeans = joblib.load("models/kmeans_model_k4.pkl")

# ------------------------------------------------------------
# Loading segment mapping (Cluster ID -> Segment Name)
# This helps show meaningful labels instead of just numbers
# ------------------------------------------------------------
with open("models/segment_map.json", "r") as f:
    segment_map = json.load(f)

# Convert JSON keys (strings) into integers for correct mapping
segment_map = {int(k): v for k, v in segment_map.items()}

# Divider for clean UI
st.markdown("---")

# Creating two columns for input and output sections
col1, col2 = st.columns([1, 1])

# ------------------------------------------------------------
# Left side: User Input Section
# ------------------------------------------------------------
with col1:
    recency = st.number_input(
        "Recency (days since last purchase)",
        min_value=0,
        value=30,
        step=1
    )

    frequency = st.number_input(
        "Frequency (number of purchases)",
        min_value=0,
        value=5,
        step=1
    )

    monetary = st.number_input(
        "Monetary (total spend)",
        min_value=0.0,
        value=1000.0,
        step=10.0
    )

    # Button triggers the prediction
    predict_btn = st.button("Predict Segment")

# ------------------------------------------------------------
# Right side: Prediction Output Section
# ------------------------------------------------------------
with col2:
    st.subheader("Prediction Output")

    # Only run prediction when the button is clicked
    if predict_btn:
        # Create input in the same format as model training (2D array)
        user_data = np.array([[recency, frequency, monetary]])

        # Scale the user input using the same scaler used during training
        user_scaled = scaler.transform(user_data)

        # Predict which cluster the user belongs to
        cluster = int(kmeans.predict(user_scaled)[0])

        # Convert predicted cluster number to segment label
        segment = segment_map.get(cluster, "Unknown")

        # Display prediction results
        st.success(f"Predicted Cluster: {cluster}")
        st.info(f"Customer Segment: {segment}")

        # ------------------------------------------------------------
        # Business recommendations based on segment
        # These are practical actions for real-world business usage
        # ------------------------------------------------------------
        if segment == "High Value":
            st.write("Recommended Action: Provide premium offers, loyalty rewards, and early-access deals.")
        elif segment == "Regular":
            st.write("Recommended Action: Cross-sell bundles and membership benefits to increase repeat purchases.")
        elif segment == "Occasional":
            st.write("Recommended Action: Offer discounts and personalized recommendations to increase engagement.")
        elif segment == "At Risk":
            st.write("Recommended Action: Run win-back campaigns, reminders, and limited-time offers.")

# Divider and footer note
st.markdown("---")
st.caption("Segmentation Model: KMeans (k=4) trained on RFM features with StandardScaler.")
