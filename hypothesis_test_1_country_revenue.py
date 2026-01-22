import pandas as pd
from scipy.stats import mannwhitneyu

# Load cleaned dataset (fast + already processed)
df = pd.read_parquet("data/online_retail_cleaned.parquet")

# We test transaction-level monetary (InvoiceNo total amount)
txn_total = df.groupby(["InvoiceNo", "Country"])["TotalPrice"].sum().reset_index()

# Filter only two countries for comparison
uk = txn_total[txn_total["Country"] == "United Kingdom"]["TotalPrice"]
germany = txn_total[txn_total["Country"] == "Germany"]["TotalPrice"]

print("UK transactions:", len(uk))
print("Germany transactions:", len(germany))

# Mann-Whitney U Test (non-parametric)
# H0: UK and Germany transaction monetary values come from same distribution
# H1: They are different
stat, p_value = mannwhitneyu(uk, germany, alternative="two-sided")

print("\nMann–Whitney U Test Results")
print("Test Statistic:", stat)
print("P-value:", p_value)

# Conclusion (alpha = 0.05)
alpha = 0.05
if p_value < alpha:
    print("\nConclusion: Reject H0")
    print("There is a statistically significant difference in transaction monetary values between UK and Germany.")
else:
    print("\nConclusion: Fail to Reject H0")
    print("No statistically significant difference found between UK and Germany transaction monetary values.")
