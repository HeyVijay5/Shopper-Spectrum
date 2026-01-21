import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Path to cleaned transaction dataset (already filtered for valid rows)
clean_path = "data/online_retail_cleaned.parquet"

# Load cleaned data for recommendation system creation
df = pd.read_parquet(clean_path)

# Select only the columns required for building the recommendation model
# We are not removing data from the dataset, only selecting what is needed for computation
df = df[["CustomerID", "Description", "Quantity"]]

# Remove rows with missing CustomerID or Description since they cannot be used for recommendations
df = df.dropna(subset=["CustomerID", "Description"])

# Clean product descriptions to avoid duplicates caused by extra spaces
df["Description"] = df["Description"].astype(str).str.strip()

# ------------------------------------------------------------
# Create Customer-Product Interaction Matrix
# Rows = customers, Columns = products, Values = quantity purchased
# This represents purchase history strength for collaborative filtering
# ------------------------------------------------------------
pivot_table = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

# Print the size of the matrix to understand the scale of computation
print("Customer-Product Matrix Shape:", pivot_table.shape)

# ------------------------------------------------------------
# Compute product-to-product cosine similarity
# Transpose is used because we want similarity between products (columns)
# ------------------------------------------------------------
product_similarity = cosine_similarity(pivot_table.T)

# Convert similarity matrix into a DataFrame for easy lookup in Streamlit
product_similarity_df = pd.DataFrame(
    product_similarity,
    index=pivot_table.columns,
    columns=pivot_table.columns
)

# ------------------------------------------------------------
# Save recommendation artifacts for Streamlit
# Saving precomputed similarity ensures fast real-time recommendations
# ------------------------------------------------------------
joblib.dump(product_similarity_df, "models/product_similarity.pkl")
joblib.dump(list(pivot_table.columns), "models/product_list.pkl")

# Confirmation output showing saved model files
print("Saved:")
print("models/product_similarity.pkl")
print("models/product_list.pkl")

# Q1. Why do we save product_similarity.pkl instead of calculating similarity live in Streamlit?
# Answer: Computing cosine similarity on the full matrix is heavy and would slow down the app.
# Precomputing and saving makes recommendations instant and improves user experience.

# Q2. Why do we save product_list.pkl separately?
# Answer: The product list is required to populate the Streamlit dropdown input safely and consistently.
# It prevents invalid product names and avoids lookup errors in the recommendation engine.

# Q3. How does cosine similarity-based recommendation improve e-commerce performance?
# Answer: It increases cross-selling by suggesting products that customers commonly buy together.
# This boosts average order value and improves shopping experience through discovery.

# Q4. How can this recommendation system support inventory and stock optimization?
# Answer: High similarity products often represent co-demand patterns and bundle opportunities.
# Businesses can stock related items together to reduce stockouts and missed revenue.