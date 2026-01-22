import joblib
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# Load similarity matrix and product list
product_similarity_df = joblib.load("models/product_similarity.pkl")
product_list = joblib.load("models/product_list.pkl")

np.random.seed(42)

top_n = 5
num_trials = 200  # number of products we test (keep moderate for laptop)

top5_scores = []
random_scores = []

print("Running hypothesis test on recommendation similarity...\n")

# Pick random sample of products to evaluate
sample_products = np.random.choice(product_list, size=min(num_trials, len(product_list)), replace=False)

for product in sample_products:
    # Get similarity scores for the chosen product
    sim_scores = product_similarity_df[product].sort_values(ascending=False)

    # Remove self-similarity
    sim_scores = sim_scores.drop(product)

    # Top 5 recommendation similarities
    top5 = sim_scores.head(top_n).values
    top5_scores.extend(top5)

    # Random 5 product similarities (baseline)
    random_items = np.random.choice(sim_scores.index, size=top_n, replace=False)
    random_vals = sim_scores.loc[random_items].values
    random_scores.extend(random_vals)

top5_scores = np.array(top5_scores)
random_scores = np.array(random_scores)

print("Top-5 similarity samples:", len(top5_scores))
print("Random similarity samples:", len(random_scores))

print("\nTop-5 Similarity Mean:", top5_scores.mean())
print("Random Similarity Mean:", random_scores.mean())

# Mann–Whitney U test (one-sided)
# H0: top5_scores <= random_scores
# H1: top5_scores > random_scores
stat, p_value = mannwhitneyu(top5_scores, random_scores, alternative="greater")

print("\nMann–Whitney U Test Results")
print("Test Statistic:", stat)
print("P-value:", p_value)

alpha = 0.05
if p_value < alpha:
    print("\nConclusion: Reject H0")
    print("The recommendation system produces significantly higher similarity than random suggestions.")
else:
    print("\nConclusion: Fail to Reject H0")
    print("The recommendation system does not significantly outperform random product suggestions.")
