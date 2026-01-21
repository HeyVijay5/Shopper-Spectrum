import pandas as pd

# Input file path (raw dataset)
input_path = "data/online_retail.csv"

# Output file path (cleaned dataset saved in Parquet format for faster loading)
output_path = "data/online_retail_cleaned.parquet"

# Load the raw CSV dataset using proper encoding (common for retail datasets)
df = pd.read_csv(input_path, encoding="ISO-8859-1")

# Print the original dataset size to verify all rows and columns are loaded
print("Original shape:", df.shape)

# Remove rows where CustomerID is missing
# CustomerID is mandatory for RFM segmentation and purchase history matrix creation
df = df.dropna(subset=["CustomerID"])

# Remove cancelled transactions
# Cancelled invoices usually start with 'C' and should not be included in analysis
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Remove invalid transactions where Quantity is zero or negative
# These records can distort monetary calculations and recommendation results
df = df[df["Quantity"] > 0]

# Remove invalid transactions where UnitPrice is zero or negative
# Negative or zero prices are not meaningful for revenue and spending analysis
df = df[df["UnitPrice"] > 0]

# Create a TotalPrice column to calculate revenue per row
# This is required for Monetary value in RFM analysis
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Print the cleaned dataset size to verify data was filtered correctly
print("Cleaned shape:", df.shape)

# Save cleaned dataset in Parquet format
# Parquet is smaller and faster than CSV, improving performance for later steps and Streamlit
df.to_parquet(output_path, index=False)

# Confirm output file location
print("Saved cleaned dataset to:", output_path)

# Q1. Why do we remove rows with missing CustomerID?
# Answer: CustomerID is required to build customer-level insights like RFM segmentation and purchase behavior history.
# Without it, transactions cannot be linked to a customer and reduce the reliability of analysis.
# Q2. Why do we save the cleaned file as Parquet instead of CSV?
# Answer: Parquet loads faster, takes less storage, and performs better for large datasets during repeated analysis.
# This makes clustering, recommendation generation, and Streamlit loading significantly smoother.
# Q3. Why is removing cancelled invoices important for business accuracy?
# Answer: Cancelled invoices do not represent real completed sales, so keeping them inflates demand and revenue incorrectly.
# Removing them ensures segmentation and inventory insights reflect true customer purchasing behavior.
# Q4. How does creating TotalPrice support real-time business use cases?
# Answer: TotalPrice enables accurate revenue and Monetary calculations, which directly drives segmentation and customer value analysis.
# It supports pricing, retention targeting, and inventory planning based on actual spending patterns.