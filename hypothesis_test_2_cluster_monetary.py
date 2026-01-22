import pandas as pd
from scipy.stats import kruskal

# Load segmented customers (KMeans output)
segments_path = "data/customer_segments.csv"
rfm = pd.read_csv(segments_path)

# Ensure correct datatype
rfm["Cluster"] = rfm["Cluster"].astype(int)

print("Clusters available:", sorted(rfm["Cluster"].unique()))

# Prepare Monetary values for each cluster
cluster_groups = []
for c in sorted(rfm["Cluster"].unique()):
    group = rfm[rfm["Cluster"] == c]["Monetary"].values
    cluster_groups.append(group)
    print(f"Cluster {c} -> Customers: {len(group)}, Avg Monetary: {group.mean():.2f}")

# Kruskal–Wallis test (non-parametric)
# H0: All clusters have the same spending distribution
# H1: At least one cluster differs
stat, p_value = kruskal(*cluster_groups)

print("\nKruskal–Wallis Test Results")
print("Test Statistic:", stat)
print("P-value:", p_value)

# Conclusion (alpha = 0.05)
alpha = 0.05
if p_value < alpha:
    print("\nConclusion: Reject H0")
    print("Customer segments have statistically significant differences in Monetary spending.")
else:
    print("\nConclusion: Fail to Reject H0")
    print("No statistically significant difference found between cluster Monetary spending.")
