import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Path to cleaned dataset (contains only valid transactions)
clean_path = "data/online_retail_cleaned.parquet"

# Load cleaned retail dataset
df = pd.read_parquet(clean_path)

# Keep only required columns for recommendation computation
# We are NOT deleting dataset columns permanently, only selecting needed columns for this model step
df = df[["CustomerID", "Description", "Quantity"]]

# Remove rows where product description is missing
# Product description is required to build the customer-product interaction matrix
df = df.dropna(subset=["Description"])

# Ensure descriptions are clean and consistent
# This avoids duplicates caused by extra spaces or mixed data types
df["Description"] = df["Description"].astype(str).str.strip()

# ------------------------------------------------------------
# Create CustomerID-Description matrix (purchase history matrix)
# Rows = customers
# Columns = products
# Values = total quantity purchased (strength of interaction)
# ------------------------------------------------------------
pivot_table = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

# Print matrix shape to confirm scale of recommendation system
print("Customer-Product Matrix Shape:", pivot_table.shape)

# ------------------------------------------------------------
# Compute cosine similarity between products
# Similarity is calculated between product vectors (columns), so we transpose pivot_table
# ------------------------------------------------------------
product_similarity = cosine_similarity(pivot_table.T)

# Convert similarity matrix into a DataFrame for easier lookup and recommendations
product_similarity_df = pd.DataFrame(
    product_similarity,
    index=pivot_table.columns,
    columns=pivot_table.columns
)

# Print confirmation and similarity matrix size
print("Product similarity matrix created.")
print("Similarity Matrix Shape:", product_similarity_df.shape)

# ------------------------------------------------------------
# Function: Recommend top N similar products based on cosine similarity
# This matches the project requirement of item-based collaborative filtering
# ------------------------------------------------------------
def recommend_products(product_name, top_n=5):
    # Handle case where product is not available in the dataset
    if product_name not in product_similarity_df.index:
        return f"Product '{product_name}' not found in dataset."

    # Sort similar products by similarity score (highest similarity first)
    similarity_scores = product_similarity_df[product_name].sort_values(ascending=False)

    # Remove the same product itself from recommendations
    similarity_scores = similarity_scores.drop(product_name)

    # Return only the top N recommended product names
    return list(similarity_scores.head(top_n).index)

# ------------------------------------------------------------
# Test recommendation output using one sample product from the dataset
# ------------------------------------------------------------
test_product = pivot_table.columns[0]
print("\nExample Product:", test_product)
print("Top 5 Recommendations:", recommend_products(test_product))

# Q1. Why do we create a CustomerID–Description pivot table for recommendations?
# Answer: The pivot table captures customer purchase history in a structured matrix format for collaborative filtering.
# It allows us to compare products based on shared buying behavior across customers.

# Q2. Why do we compute cosine similarity on pivot_table.T and not directly on pivot_table?
# Answer: We want similarity between products, so products must be treated as vectors (columns).
# Transposing makes each product a vector of customer purchase quantities for accurate similarity scoring.

# Q3. Why is item-based collaborative filtering useful for e-commerce recommendations?
# Answer: It recommends products based on real co-purchase patterns rather than manual product tagging.
# This improves personalization, cross-selling, and customer shopping experience.

# Q4. What business value does “Top 5 similar products” recommendation provide?
# Answer: It increases average order value by encouraging customers to add related items to their cart.
# It also improves product discovery, reducing bounce rate and increasing conversions.