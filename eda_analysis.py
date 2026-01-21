# Graph 1: Top 10 Countries by Number of Transactions
# Insight: The United Kingdom dominates transaction volume, while other countries contribute relatively small order counts.
# Business benefit: This helps prioritize UK-first marketing, inventory planning, and customer support scaling before expanding globally.

# Graph 2: Top 10 Selling Products (by Quantity)
# Insight: A small set of products drives a very large portion of total quantity sold, showing clear bestsellers.
# Business benefit: These products should be prioritized for stock availability, bundle offers, and recommendations to boost conversions.

# Graph 3: Daily Sales Trend
# Insight: Sales fluctuate heavily day-to-day with noticeable peaks, indicating seasonal spikes or bulk-order events.
# Business benefit: Supports demand forecasting, campaign timing, and proactive inventory restocking around peak periods.

# Graph 4: Monetary Distribution per Transaction
# Insight: Most transactions are low-value with a long tail of very high-value orders (right-skewed distribution).
# Business benefit: Helps define pricing thresholds, detect unusually large orders, and design minimum-cart offers to increase AOV.

# Graph 5: Monetary Distribution per Customer
# Insight: Most customers spend small amounts, while a few customers contribute extremely high lifetime spend (VIP pattern).
# Business benefit: Enables VIP retention and loyalty targeting since a small customer group likely drives a disproportionate share of revenue.

# Graph 6: Recency Distribution (RFM)
# Insight: Many customers purchased recently, but there is a long tail of customers inactive for a long time.
# Business benefit: Helps identify churn risk and prioritize “At Risk” win-back campaigns using Recency thresholds.

# Graph 7: Frequency Distribution (RFM)
# Insight: Most customers buy only a few times, while a small group purchases repeatedly at high frequency.
# Business benefit: Supports loyalty program design to convert one-time buyers into repeat customers and protect repeat purchasers.

# Graph 8: Monetary Distribution (RFM)
# Insight: Monetary value is highly right-skewed, showing a large base of low spenders and few high spenders.
# Business benefit: Helps build segment-driven strategy like premium upsell for high spenders and discounts for low spenders.

import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned transaction dataset and the customer-level RFM table
clean_path = "data/online_retail_cleaned.parquet"
rfm_path = "data/rfm_table.csv"

df = pd.read_parquet(clean_path)
rfm = pd.read_csv(rfm_path)

# Convert InvoiceDate to datetime format for time-series analysis
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Print shapes for validation to ensure data is loaded correctly
print("Cleaned dataset shape:", df.shape)
print("RFM table shape:", rfm.shape)

# ------------------------------------------------------------
# 1) Transaction volume by country
# Helps identify which geographies contribute the most business activity
# ------------------------------------------------------------
country_txn = df.groupby("Country")["InvoiceNo"].nunique().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
plt.bar(country_txn.index, country_txn.values)
plt.title("Top 10 Countries by Number of Transactions")
plt.xlabel("Country")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("data/eda_country_transactions.png")
plt.show()

# ------------------------------------------------------------
# 2) Top-selling products (by quantity)
# Helps identify bestsellers that drive high demand and fast inventory movement
# ------------------------------------------------------------
top_products = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
plt.barh(top_products.index[::-1], top_products.values[::-1])
plt.title("Top 10 Selling Products (by Quantity)")
plt.xlabel("Total Quantity Sold")
plt.ylabel("Product Description")
plt.tight_layout()
plt.savefig("data/eda_top_products.png")
plt.show()

# ------------------------------------------------------------
# 3) Purchase trends over time (daily sales)
# Reveals seasonality, peaks, and demand patterns across time
# ------------------------------------------------------------
daily_sales = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()

plt.figure(figsize=(12, 5))
plt.plot(daily_sales.index, daily_sales.values)
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("data/eda_daily_sales_trend.png")
plt.show()

# ------------------------------------------------------------
# 4) Monetary distribution per transaction
# Shows whether orders are mostly small-value or have many large outliers
# ------------------------------------------------------------
txn_monetary = df.groupby("InvoiceNo")["TotalPrice"].sum()

plt.figure(figsize=(8, 5))
plt.hist(txn_monetary, bins=50)
plt.title("Monetary Distribution per Transaction")
plt.xlabel("Transaction Total Amount")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/eda_transaction_monetary_dist.png")
plt.show()

# ------------------------------------------------------------
# 5) Monetary distribution per customer
# Shows customer spend inequality (many small spenders vs few VIP spenders)
# ------------------------------------------------------------
customer_monetary = df.groupby("CustomerID")["TotalPrice"].sum()

plt.figure(figsize=(8, 5))
plt.hist(customer_monetary, bins=50)
plt.title("Monetary Distribution per Customer")
plt.xlabel("Customer Total Spend")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/eda_customer_monetary_dist.png")
plt.show()

# ------------------------------------------------------------
# 6) RFM distributions
# Visualizing Recency, Frequency, Monetary helps define customer behaviors and segments
# ------------------------------------------------------------

# Recency distribution: days since last purchase
plt.figure(figsize=(8, 5))
plt.hist(rfm["Recency"], bins=50)
plt.title("Recency Distribution")
plt.xlabel("Recency (days)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/rfm_recency_dist.png")
plt.show()

# Frequency distribution: number of purchases
plt.figure(figsize=(8, 5))
plt.hist(rfm["Frequency"], bins=50)
plt.title("Frequency Distribution")
plt.xlabel("Frequency (unique invoices)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/rfm_frequency_dist.png")
plt.show()

# Monetary distribution: total spend across the dataset
plt.figure(figsize=(8, 5))
plt.hist(rfm["Monetary"], bins=50)
plt.title("Monetary Distribution")
plt.xlabel("Monetary Value (total spend)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/rfm_monetary_dist.png")
plt.show()

# Final message confirming all plots were generated
print("EDA plots saved inside data/ folder")


# Code Questions (EDA Script)

# Q1. Why do we convert InvoiceDate to datetime using pd.to_datetime()?
# Answer: It enables time-based grouping like daily, monthly, or weekly trends accurately.
# Without datetime conversion, trend analysis and forecasting become unreliable.
# Q2. Why do we use nunique() on InvoiceNo for country transactions instead of counting rows?
# Answer: nunique() counts distinct invoices, which better represents actual transaction volume.
# Row count can be misleading because one invoice can have multiple product line-items.
# Q3. Why do we save plots using plt.savefig() before plt.show()?
# Answer: Saving ensures plots are stored for reports and submission even if the window is closed.
# It builds reproducible outputs that are easy to attach in documentation.
# Q4. Why are we grouping daily sales by TotalPrice.sum() instead of Quantity?
# Answer: Revenue trend is more meaningful for business performance tracking than units sold alone.
# It reflects actual monetary impact and supports pricing and forecasting decisions.

# Graph Questions (EDA Interpretation)

# Q5. What does the huge dominance of the UK in transactions tell us?
# Answer: The UK is the primary market and drives most operational workload and revenue opportunity.
# It should be prioritized for targeting, fulfillment optimization, and customer experience improvements.
# Q6. Why is the transaction monetary distribution heavily right-skewed?
# Answer: Most customers place small orders while a few orders are exceptionally large (bulk purchases).
# This is typical in retail and supports identifying high-value customers and fraud/anomaly checks.
# Q7. What do the peaks in daily sales indicate in the trend plot?
# Answer: Peaks usually represent seasonal events, promotions, or bulk buying periods.
# This helps schedule inventory restocking and campaign timing more accurately.
# Q8. Why are Frequency and Monetary distributions important before clustering?
# Answer: They show that customer behavior is not uniform and has strong outliers and long tails.
# This justifies scaling and segmentation to make customer strategies more precise.

# Business Questions (Use Case Alignment)

# Q9. How does country-level transaction analysis support business growth?
# Answer: It helps identify where demand is strongest and where expansion opportunities exist.
# It improves decision-making for logistics, marketing spend, and localization strategy.
# Q10. How do top-selling products support personalized recommendation design?
# Answer: Bestsellers become strong fallback recommendations for new or low-history customers.
# They also help create bundles and boost cross-sell conversion rates.
# Q11. How does RFM recency distribution connect to retention programs?
# Answer: Customers with high recency are inactive and most likely to churn.
# Retention campaigns can target them with win-back offers before they are lost permanently.
# Q12. How does customer monetary distribution support dynamic pricing strategy?
# Answer: High spenders are often less price-sensitive, while low spenders respond to discounts.
# This enables segment-based pricing and promotions without sacrificing overall revenue.