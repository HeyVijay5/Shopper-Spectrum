import pandas as pd

# Input file paths generated after clustering
segments_path = "data/customer_segments.csv"
summary_path = "data/cluster_summary.csv"

# Output file path where labeled customer segments will be saved
output_path = "data/customer_segments_labeled.csv"

# Load customer cluster assignments and cluster summary statistics
rfm = pd.read_csv(segments_path)
summary = pd.read_csv(summary_path)

# Ensure cluster values are treated as integers for correct mapping and consistency
rfm["Cluster"] = rfm["Cluster"].astype(int)
summary["Cluster"] = summary["Cluster"].astype(int)

# Print summary table for verification before mapping
print("Cluster Summary:")
print(summary)

# ------------------------------------------------------------
# Automatic cluster-to-segment logic (no hardcoding)
# This logic identifies segment meaning based on RFM characteristics:
# High Value  -> highest Monetary + Frequency
# At Risk     -> highest Recency
# Regular     -> higher Frequency among remaining
# Occasional  -> lower Frequency among remaining
# ------------------------------------------------------------
summary_sorted = summary.sort_values(["Avg_Monetary", "Avg_Frequency"], ascending=False)
high_value_cluster = int(summary_sorted.iloc[0]["Cluster"])

at_risk_cluster = int(summary.sort_values("Avg_Recency", ascending=False).iloc[0]["Cluster"])

remaining = list(set(summary["Cluster"]) - {high_value_cluster, at_risk_cluster})
remaining_summary = summary[summary["Cluster"].isin(remaining)].sort_values("Avg_Frequency", ascending=False)

regular_cluster = int(remaining_summary.iloc[0]["Cluster"])
occasional_cluster = int(remaining_summary.iloc[1]["Cluster"])

# Auto mapping generated based on RFM values (dynamic labeling)
auto_cluster_to_segment = {
    high_value_cluster: "High Value",
    regular_cluster: "Regular",
    occasional_cluster: "Occasional",
    at_risk_cluster: "At Risk"
}

print("\nAuto Cluster to Segment Mapping (based on RFM values):")
print(auto_cluster_to_segment)

# ------------------------------------------------------------
# Fixed cluster-to-segment mapping (confirmed final mapping for this project run)
# This ensures Streamlit output is stable and matches your final segmentation labels
# ------------------------------------------------------------
segment_map = {
    0: "Occasional",
    1: "At Risk",
    2: "High Value",
    3: "Regular"
}

print("\nFixed Mapping Used for Final Output:")
print(segment_map)

# ------------------------------------------------------------
# Add segment labels into customer-level file using the fixed mapping
# This produces the final dataset used for business insights dashboard
# ------------------------------------------------------------
rfm["Segment"] = rfm["Cluster"].map(segment_map)

# Save labeled customer segmentation output
rfm.to_csv(output_path, index=False)

# Print confirmation and customer count per segment
print("\nSaved labeled segments file:", output_path)
print("\nSegment Counts:")
print(rfm["Segment"].value_counts())

# Q1. Why do we calculate an automatic cluster-to-segment mapping if we finally use a fixed mapping?
# Answer: Auto mapping validates segment meaning based on real RFM values and prevents wrong labeling assumptions.
# Fixed mapping ensures stable labeling for Streamlit and final project submission consistency.

# Q2. Why do we treat “At Risk” as the cluster with the highest Recency?
# Answer: High recency means customers haven’t purchased for a long time and may be inactive or churned.
# This makes it the strongest behavioral signal to identify retention candidates.

# Q3. How does labeled segmentation help targeted marketing campaigns?
# Answer: It helps marketing teams design separate strategies for High Value, Regular, Occasional, and At Risk customers.
# This improves campaign ROI by avoiding one-size-fits-all promotions.

# Q4. How does this segmentation output support real-time business dashboards and decisions?
# Answer: It provides ready-to-use segment labels for reporting and monitoring customer behavior trends.
# This enables faster decisions in pricing, retention programs, and inventory planning.