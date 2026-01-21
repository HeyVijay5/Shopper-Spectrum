# Q1. Why do we use custom CSS inside st.markdown() in the homepage?
# Answer: CSS improves the UI by adding clean typography, spacing, and card-based layout for better readability.
# It makes the project look more polished and product-ready instead of a basic prototype.
# Q2. Why are we using st.columns(3) on the homepage?
# Answer: A three-column layout clearly separates the project into segmentation, recommendation, and business insights modules.
# This improves navigation clarity and keeps the user experience structured.
# Q3. Why is it important to show the three main modules on the homepage?
# Answer: It communicates the end-to-end business value of the project in one view for reviewers and stakeholders.
# It also helps users understand how segmentation and recommendations connect to decision-making.
# Q4. How does a strong UI/UX homepage improve the business impact of the Streamlit app?
# Answer: A clean dashboard-style entry page builds trust and makes insights easier to explore for non-technical users.
# It increases adoption and makes the system feel like a deployable analytics product.

import streamlit as st

# Configure the Streamlit page settings such as title, icon, and layout width
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛍️",
    layout="wide"
)

# Adding custom CSS styling to make the homepage look more professional
# This controls font sizes, colors, and card design for a clean UI/UX
st.markdown(
    """
    <style>
    .main-title {
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 18px;
        color: #6b7280;
        margin-top: 0px;
    }
    .card {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main title and subtitle for the homepage
st.markdown('<div class="main-title">Shopper Spectrum</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Customer Segmentation and Product Recommendation System</div>', unsafe_allow_html=True)

# Adding spacing for better layout
st.write("")
st.write("")

# Creating a 3-column layout to highlight the main modules of the project
col1, col2, col3 = st.columns(3)

with col1:
    # Card 1: Customer Segmentation module overview
    st.markdown(
        """
        <div class="card">
            <h3>Customer Segmentation</h3>
            <p>Segment customers using RFM analysis and KMeans clustering for targeted marketing strategies.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    # Card 2: Product Recommendation module overview
    st.markdown(
        """
        <div class="card">
            <h3>Product Recommendation</h3>
            <p>Recommend 5 similar products using item-based collaborative filtering and cosine similarity.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    # Card 3: Business Insights module overview
    st.markdown(
        """
        <div class="card">
            <h3>Business Insights</h3>
            <p>Enable retention strategies, dynamic pricing, and inventory optimization using customer behavior insights.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Extra spacing and navigation guidance for the user
st.write("")
st.info("Use the sidebar to navigate: Customer Segmentation and Product Recommendation.")

# Cluster Overview (What each cluster represents)
# Cluster 2 (13 customers) = High Value (VIP)

# Lowest Recency (7.38 days), highest Frequency (82.54), extremely high Monetary (127,338)

# These are your most loyal and highest revenue customers, even though they are very few.

# Cluster 3 (204 customers) = Regular / Loyal Customers

# Low Recency (15.5), strong Frequency (22.33), high Monetary (12,709)

# This segment drives stable repeat revenue and is ideal for upselling and bundles.

# Cluster 0 (3054 customers) = Occasional / Low-Mid Value Customers

# Medium Recency (43.7), low Frequency (3.68), low Monetary (1,359)

# Largest customer base, but spending is relatively low per customer.

# Cluster 1 (1067 customers) = At Risk Customers

# Very high Recency (248 days), very low Frequency (1.55), lowest Monetary (480)

# This segment is disengaged and has high churn probability.

# Business Insights (High-impact points for PPT/report)
# 1) Revenue is concentrated in a small customer group

# Cluster 2 has only 13 customers but the highest spend by a huge margin, showing classic “VIP concentration” behavior.
# This proves why retention and premium rewards for top customers directly protect revenue.

# 2) Cluster 1 is your churn risk segment

# High Recency + low Frequency means these customers haven’t purchased in a long time and may be lost.
# Retention campaigns should prioritize Cluster 1 because it’s easier to win back than acquiring new customers.

# 3) Cluster 0 is your growth opportunity segment

# It’s the biggest cluster, so even a small improvement in frequency can create major revenue impact.
# Best strategy here is personalized recommendations, discounts, and bundles to increase repeat orders.

# 4) Cluster 3 is your “most scalable profit segment”

# They already purchase frequently and recently, so cross-selling and loyalty programs will increase AOV easily.
# This segment is your best target for “recommended products” and membership-driven growth.

# Real-time Business Use Case Mapping

# Targeted Marketing: Different campaigns for VIP, Regular, Occasional, At Risk segments

# Recommendations: Boost conversions for Cluster 0 and Cluster 3 with top-5 similar products

# Retention: Win-back strategy specifically for Cluster 1

# Dynamic Pricing: Avoid heavy discounts for Cluster 2, discounts for Cluster 0 activation

# Inventory: Prioritize stock for items frequently purchased by Cluster 2 and Cluster 3 customers