import pandas as pd

# File path for the raw dataset stored inside the data folder
file_path = "data/online_retail.csv"

# Load the dataset using the correct encoding to avoid unreadable characters
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Basic confirmation that the dataset is loaded correctly
print("Dataset Loaded Successfully")

# Print dataset size to confirm all rows and columns are imported
print("Rows, Columns:", df.shape)

# Display the column names to understand the dataset structure
print("\nColumn Names:")
print(df.columns)

# Preview the first 5 rows to verify data looks correct and readable
print("\nFirst 5 rows:")
print(df.head())


# Q1. Why do we use encoding="ISO-8859-1" while loading the dataset?
# Answer: Many retail datasets include special characters that may break default UTF-8 decoding.
# This encoding ensures the file loads correctly without errors or corrupted text.

# Q2. Why do we print df.shape, column names, and df.head() after loading?
# Answer: It validates that the dataset is fully loaded and the structure matches expectations.
# This prevents silent data issues before cleaning, EDA, clustering, and recommendations.

# Q3. How does this loading validation step help reduce project failures later?
# Answer: It detects missing columns, incorrect formats, or corrupted records early in the pipeline.
# Early verification saves time and prevents downstream debugging during model building.

# Q4. Why is it important to check raw data before business insights and recommendation modeling?
# Answer: Business decisions depend on clean and accurate input data, so verification is mandatory.
# Incorrect loading can lead to wrong segmentation, misleading insights, and poor recommendations.