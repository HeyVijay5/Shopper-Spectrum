import pandas as pd

# Path to cleaned dataset (after removing missing customers, cancellations, invalid values)
clean_path = "data/online_retail_cleaned.parquet"

# Output path where the RFM table will be saved for clustering and segmentation
rfm_output_path = "data/rfm_table.csv"

# Load cleaned data from parquet for faster performance on large datasets
df = pd.read_parquet(clean_path)

# Print dataset shape to confirm data is loaded correctly
print("Loaded cleaned data:", df.shape)

# Convert InvoiceDate to datetime format for accurate time-based calculations
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# ------------------------------------------------------------
# Reference date for Recency calculation
# Using (max date + 1 day) ensures the most recent purchase has recency = 1 day or 0 days depending on calculation
# This keeps recency values consistent and comparable across customers
# ------------------------------------------------------------
reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# ------------------------------------------------------------
# Build RFM features for each customer:
# Recency   = days since last purchase
# Frequency = count of unique invoices (number of purchases)
# Monetary  = total spend across all purchases
# ------------------------------------------------------------
rfm = df.groupby("CustomerID").agg(
    Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("TotalPrice", "sum")
).reset_index()

# Print RFM table size and preview to verify correct output
print("RFM table shape:", rfm.shape)
print(rfm.head())

# Save RFM table for clustering and Streamlit usage
rfm.to_csv(rfm_output_path, index=False)

# Confirmation message showing file output location
print("Saved RFM table to:", rfm_output_path)

# Q1. Why do we use reference_date = max(InvoiceDate) + 1 day for Recency calculation?
# Answer: It standardizes recency so every customer is measured relative to the latest transaction date in the dataset.
# This ensures recency values are consistent and meaningful across all customers.

# Q2. Why is Frequency calculated using nunique(InvoiceNo) instead of counting rows?
# Answer: One invoice can contain multiple product rows, so counting rows overestimates purchases.
# Unique invoices correctly represent the number of transactions made by the customer.

# Q3. Why is RFM a strong foundation for customer segmentation?
# Answer: RFM captures customer engagement (Recency, Frequency) and customer value (Monetary) in a simple structure.
# It produces interpretable segments that business teams can act on immediately.

# Q4. How does this RFM table support real-time business use cases like retention and dynamic pricing?
# Answer: It identifies which customers are inactive, loyal, or high spenders for targeted strategies.
# This enables smarter pricing, personalized campaigns, and churn prevention based on behavior.