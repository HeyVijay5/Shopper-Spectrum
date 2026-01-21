# 1. Why are we loading product_similarity.pkl and product_list.pkl instead of computing cosine similarity inside Streamlit?
# Answer: Precomputing and saving the similarity matrix makes recommendations instant and keeps the app responsive.
# It avoids heavy matrix computation on every user click, which is important for large datasets.
# Q2. Why do we drop the same product from similarity scores before selecting the top 5 recommendations?
# Answer: The input product always has the highest similarity score with itself (score = 1), so it must be removed.
# This ensures the output contains only genuinely related products instead of repeating the same item.

# Business Insights (Why this method?)

# Q3. Why did we use Item-based Collaborative Filtering with cosine similarity for recommendations instead of content-based filtering?
# Answer: It recommends products based on real customer purchase behavior, which captures strong co-buying patterns.
# It also works well without needing product categories or detailed attributes, making it scalable and practical.
# Q4. How does this recommendation system support real-time business goals like cross-selling and inventory optimization?
# Answer: Similar-item recommendations increase cart value by suggesting products frequently purchased together.
# It also highlights high-demand item relationships, helping businesses plan bundles and optimize stock availability.

import streamlit as st
import joblib

# Page configuration for better layout and page title
st.set_page_config(page_title="Product Recommendation", layout="wide")

# Page heading and short description
st.title("Product Recommendation")
st.write("Select a product name and get 5 similar recommendations based on cosine similarity.")

# ------------------------------------------------------------
# Load precomputed similarity matrix and product list
# This avoids expensive computation inside Streamlit
# ------------------------------------------------------------
product_similarity_df = joblib.load("models/product_similarity.pkl")
product_list = joblib.load("models/product_list.pkl")

# ------------------------------------------------------------
# Custom CSS for card-style recommendation output
# Improves UI/UX and makes recommendations look professional
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .rec-card {
        background-color: #ffffff;
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 10px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        font-size: 16px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Divider for better structure
st.markdown("---")

# Creating two columns: input selection (left) and output (right)
col1, col2 = st.columns([1, 1])

# ------------------------------------------------------------
# Left side: Product selection and button
# ------------------------------------------------------------
with col1:
    # Dropdown makes input safe and avoids spelling errors
    product_name = st.selectbox("Select Product Name", product_list)

    # Button triggers recommendation generation
    recommend_btn = st.button("Get Recommendations")

# ------------------------------------------------------------
# Right side: Recommendation list display
# ------------------------------------------------------------
with col2:
    st.subheader("Top 5 Recommended Products")

    # Only generate recommendations when button is clicked
    if recommend_btn:
        # Get similarity scores for selected product
        similarity_scores = product_similarity_df[product_name].sort_values(ascending=False)

        # Remove the same product from its recommendation list
        similarity_scores = similarity_scores.drop(product_name)

        # Select top 5 most similar products
        recommendations = list(similarity_scores.head(5).index)

        # Display results in clean card format
        for i, item in enumerate(recommendations, start=1):
            st.markdown(f"<div class='rec-card'>{i}. {item}</div>", unsafe_allow_html=True)

# Divider and footer note
st.markdown("---")
st.caption("Recommendation Engine: Item-based Collaborative Filtering using Cosine Similarity.")
