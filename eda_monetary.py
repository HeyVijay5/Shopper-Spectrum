# Plot 1: Transaction Monetary Distribution (log1p scale)
# Insight: After log transformation, transaction values form a clear bell-like spread, confirming heavy right-skew in raw order totals.
# Business benefit: Helps set practical order-value bands for discounting, fraud checks, and average order value (AOV) improvement plans.

# Plot 2: Customer Monetary Distribution (log1p scale)
# Insight: Customer spend shows strong spread even after log scale, meaning customer value varies widely across the base.
# Business benefit: Supports VIP identification and customer tiering strategies for retention and premium loyalty programs.

# Plot 3: Customer Monetary Distribution (Up to 99th percentile)
# Insight: Even excluding extreme outliers visually, most customers still lie in the low-spend region with a long tail.
# Business benefit: Helps design “mass-market” campaigns for low spenders while protecting high spenders with separate VIP treatment.

# Plot 4: Transaction Monetary Distribution (Up to 99th percentile)
# Insight: The majority of transactions fall into a small monetary range, indicating typical cart sizes are consistent and predictable.
# Business benefit: Useful for inventory planning and optimizing cart-level upsell strategies like “add-on products” and bundles.

# Plot 5: Cumulative Revenue Contribution by Customers (Pareto Curve)
# Insight: The curve rises very fast initially, proving a small percentage of customers contribute a large share of revenue (Pareto behavior).
# Business benefit: Justifies prioritizing retention and personalized experiences for top spenders because losing them impacts revenue disproportionately.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the cleaned dataset saved earlier (contains valid transactions + TotalPrice)
clean_path = "data/online_retail_cleaned.parquet"
df = pd.read_parquet(clean_path)

# ------------------------------------------------------------
# Transaction-level monetary analysis
# Grouping by InvoiceNo gives total spend per transaction (order/cart value)
# ------------------------------------------------------------
txn_monetary = df.groupby("InvoiceNo")["TotalPrice"].sum()

# ------------------------------------------------------------
# Customer-level monetary analysis
# Grouping by CustomerID gives total spend per customer (customer lifetime value proxy)
# ------------------------------------------------------------
cust_monetary = df.groupby("CustomerID")["TotalPrice"].sum()

# Print basic descriptive statistics for transaction values
print("Transaction Monetary Summary")
print(txn_monetary.describe())

# Print basic descriptive statistics for customer total spend
print("\nCustomer Monetary Summary")
print(cust_monetary.describe())

# ------------------------------------------------------------
# Percentile summary to understand spending distribution
# Percentiles are more informative than mean because of extreme outliers
# ------------------------------------------------------------
percentiles = [0.50, 0.75, 0.90, 0.95, 0.99]

txn_pct = txn_monetary.quantile(percentiles)
cust_pct = cust_monetary.quantile(percentiles)

print("\nTransaction Monetary Percentiles")
print(txn_pct)

print("\nCustomer Monetary Percentiles")
print(cust_pct)

# ------------------------------------------------------------
# Plot 1: Transaction monetary distribution in log1p scale
# log1p reduces skewness and makes the distribution easier to visualize
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.hist(np.log1p(txn_monetary), bins=60)
plt.title("Transaction Monetary Distribution (log1p scale)")
plt.xlabel("log1p(Transaction Total Amount)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/eda_txn_monetary_log.png")
plt.show()

# ------------------------------------------------------------
# Plot 2: Customer monetary distribution in log1p scale
# Helps visualize customer value spread after reducing extreme skew
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.hist(np.log1p(cust_monetary), bins=60)
plt.title("Customer Monetary Distribution (log1p scale)")
plt.xlabel("log1p(Customer Total Spend)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/eda_customer_monetary_log.png")
plt.show()

# ------------------------------------------------------------
# Plot 3: Trimmed view up to 99th percentile (visual clarity only)
# This does NOT delete data, it is only used to view common ranges better
# ------------------------------------------------------------
txn_limit = txn_monetary.quantile(0.99)
cust_limit = cust_monetary.quantile(0.99)

plt.figure(figsize=(9, 5))
plt.hist(txn_monetary[txn_monetary <= txn_limit], bins=60)
plt.title("Transaction Monetary Distribution (Up to 99th percentile)")
plt.xlabel("Transaction Total Amount")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/eda_txn_monetary_trimmed_99.png")
plt.show()

plt.figure(figsize=(9, 5))
plt.hist(cust_monetary[cust_monetary <= cust_limit], bins=60)
plt.title("Customer Monetary Distribution (Up to 99th percentile)")
plt.xlabel("Customer Total Spend")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("data/eda_customer_monetary_trimmed_99.png")
plt.show()

# ------------------------------------------------------------
# Plot 4: Pareto / Revenue contribution insights
# Sorting customers by spend shows how much revenue top customers contribute
# ------------------------------------------------------------
cust_sorted = cust_monetary.sort_values(ascending=False)
total_revenue = cust_sorted.sum()

top_10_share = cust_sorted.head(10).sum() / total_revenue * 100
top_50_share = cust_sorted.head(50).sum() / total_revenue * 100
top_100_share = cust_sorted.head(100).sum() / total_revenue * 100

print("\nRevenue Contribution Insights")
print(f"Top 10 customers contribute:  {top_10_share:.2f}% of total revenue")
print(f"Top 50 customers contribute:  {top_50_share:.2f}% of total revenue")
print(f"Top 100 customers contribute: {top_100_share:.2f}% of total revenue")

# Cumulative revenue curve (Pareto curve)
cum_revenue = cust_sorted.cumsum() / total_revenue

plt.figure(figsize=(9, 5))
plt.plot(range(1, len(cum_revenue) + 1), cum_revenue.values)
plt.title("Cumulative Revenue Contribution by Customers (Pareto Curve)")
plt.xlabel("Number of Customers (sorted by spend)")
plt.ylabel("Cumulative Revenue Share")
plt.tight_layout()
plt.savefig("data/eda_pareto_curve.png")
plt.show()

# Print confirmation of all saved plot outputs
print("\nSaved plots in data/ folder:")
print("eda_txn_monetary_log.png")
print("eda_customer_monetary_log.png")
print("eda_txn_monetary_trimmed_99.png")
print("eda_customer_monetary_trimmed_99.png")
print("eda_pareto_curve.png")

# Q1. Why do we use groupby("InvoiceNo")["TotalPrice"].sum() for transaction monetary?
# Answer: It calculates true order/cart value by summing all line items in the invoice.
# This gives accurate AOV patterns instead of looking at item-level prices.
# Q2. Why do we use percentiles instead of only mean/standard deviation?
# Answer: Monetary values have extreme outliers, so mean gets biased upward.
# Percentiles describe typical spend behavior more reliably for business actions.
# Q3. Why do we apply np.log1p() before plotting monetary distributions?
# Answer: Log scaling reduces skewness and makes dense spending regions visible.
# It helps detect patterns that are hidden when values are dominated by outliers.
# Q4. Why do we create a trimmed 99th percentile view if we are not deleting data?
# Answer: It improves interpretability by focusing on the majority behavior range.
# It helps communicate insights clearly without removing important high-value customers.

# Plot Interpretation Questions

# Q5. What does the log-scale transaction monetary histogram tell us?
# Answer: Most orders fall into a typical value range, with fewer high-value purchases.
# This confirms a stable core customer spending behavior with occasional large baskets.
# Q6. What does the trimmed transaction plot reveal that the raw histogram hides?
# Answer: It highlights the true shape of common order values without extreme outliers.
# This helps define realistic pricing thresholds and minimum-cart strategies.
# Q7. What does the customer monetary distribution indicate about customer inequality?
# Answer: Most customers are low spenders, and only a small fraction are heavy spenders.
# This supports segmentation strategies and tiered loyalty benefits.
# Q8. How do we interpret the Pareto curve in retail revenue?
# Answer: Revenue accumulates quickly from the top customers, showing heavy concentration.
# This proves VIP retention produces higher ROI than broad discounts for everyone.

# Business Use-Case Questions

# Q9. How does this analysis support targeted marketing campaigns?
# Answer: It separates low spenders from high spenders and helps personalize incentives.
# This avoids wasting discounts on customers who would buy anyway.
# Q10. How does this monetary analysis improve retention programs?
# Answer: High-value customers can be proactively protected with premium service and perks.
# At-risk high spenders can be flagged early to prevent major revenue drop.
# Q11. How does this help in dynamic pricing strategies?
# Answer: It reveals typical order size bands, helping decide where discounts increase conversions.
# Pricing can be optimized by segment sensitivity rather than blanket markdowns.
# Q12. How does this support inventory stock optimization?
# Answer: Knowing common cart values and high-value spikes helps forecast demand patterns.
# It reduces stockouts during peak-buy events and avoids overstock in low-demand periods.