# Q1. Why do we load online_retail_cleaned.parquet and customer_segments_labeled.csv instead of using the raw CSV directly?
# Answer: The cleaned parquet file loads faster and contains validated transactional records, which improves dashboard performance.
# # The labeled segments file connects clustering outputs to customer-level insights, making the KPIs meaningful and consistent.
# Q2. Why are KPI metrics like total revenue, customers, transactions, and products important on the dashboard homepage?
# Answer: These metrics provide an immediate high-level snapshot of business scale and performance for quick decision-making.
# They also help validate that the dataset and pipeline are working correctly before deeper analysis.

# Business Insights (Why this dashboard and use cases?)

# Q3. How does showing “Customer Segment Distribution” help the business take real actions?
# Answer: It reveals which customer groups dominate the customer base and helps prioritize marketing and retention budgets.
# For example, a high count of “At Risk” customers signals urgent churn prevention campaigns.
# Q4. Why do we include business use cases like dynamic pricing and inventory optimization in the insights page?
# Answer: It proves that the analytics pipeline is not just technical, but directly supports revenue, retention, and cost control.
# These use cases convert customer behavior patterns into practical strategies that stakeholders can implement immediately.

import streamlit as st
import pandas as pd

# Page configuration for better layout and page title
st.set_page_config(page_title="Business Insights", layout="wide")

# Page heading and short description
st.title("Business Insights Dashboard")
st.write("Key insights and real-time use cases derived from customer behavior and purchase patterns.")

# Divider for better structure
st.markdown("---")

# ------------------------------------------------------------
# Load cleaned transaction data and labeled customer segments
# Using cleaned parquet improves speed and ensures valid transactions
# ------------------------------------------------------------
df = pd.read_parquet("data/online_retail_cleaned.parquet")
segments = pd.read_csv("data/customer_segments_labeled.csv")

# ------------------------------------------------------------
# Basic KPIs for business overview
# These metrics summarize the dataset at a glance
# ------------------------------------------------------------
total_customers = segments["CustomerID"].nunique()
total_transactions = df["InvoiceNo"].nunique()
total_products = df["Description"].nunique()
total_revenue = df["TotalPrice"].sum()

# Display KPIs using Streamlit metrics in 4 columns
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Total Transactions", f"{total_transactions:,}")
col3.metric("Unique Products", f"{total_products:,}")
col4.metric("Total Revenue", f"{total_revenue:,.2f}")

st.markdown("---")

# ------------------------------------------------------------
# Segment distribution table
# Helps understand how customers are spread across segments
# ------------------------------------------------------------
st.subheader("Customer Segment Distribution")

segment_counts = segments["Segment"].value_counts().reset_index()
segment_counts.columns = ["Segment", "Customers"]

# Display segment distribution in a clean table
st.dataframe(segment_counts, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------
# Real-world business use cases based on insights from this project
# These are the key outputs expected in the project submission
# ------------------------------------------------------------
st.subheader("Real-time Business Use Cases")

with st.expander("1. Customer Segmentation for Targeted Marketing Campaigns"):
    st.write(
        """
        - High Value: premium loyalty rewards, exclusive offers, personalized bundles  
        - Regular: cross-sell campaigns, membership benefits, repeat-purchase reminders  
        - Occasional: discounts, trending products, engagement campaigns  
        - At Risk: win-back offers, reminders, limited-time coupons  
        """
    )

with st.expander("2. Personalized Product Recommendations on E-Commerce Platforms"):
    st.write(
        """
        - Recommend 5 similar products using cosine similarity  
        - Improves customer experience and increases average order value  
        - Helps cross-selling and product discovery  
        """
    )

with st.expander("3. Identifying At-Risk Customers for Retention Programs"):
    st.write(
        """
        - At Risk customers have high recency and low frequency  
        - Retention actions: reminder emails, personalized discounts, win-back campaigns  
        - Prevents churn and improves repeat purchases  
        """
    )

with st.expander("4. Dynamic Pricing Strategies Based on Purchase Behavior"):
    st.write(
        """
        - High demand products can maintain price with minimal discounts  
        - Low demand products can be promoted with discounts to increase sales  
        - Segment-based offers improve revenue without over-discounting  
        """
    )

with st.expander("5. Inventory Management and Stock Optimization"):
    st.write(
        """
        - Identify top-selling products and demand patterns  
        - Reduce overstock by forecasting low-moving items  
        - Prioritize inventory for products frequently purchased by high-value customers  
        """
    )

st.markdown("---")
st.caption("This dashboard connects clustering insights, purchase trends, and recommendations to real business decision-making.")
