# Shopper Spectrum 
Customer Segmentation & Product Recommendation System (RFM + Clustering + Cosine Similarity)

## Project Overview
**Shopper Spectrum** is an end-to-end data science project built on a large-scale retail transactions dataset (**~541,909 rows**).  
The project delivers two core business solutions:
1. **Customer Segmentation** using **RFM Analysis + Clustering**
2. **Product Recommendation System** using **Item-Based Collaborative Filtering + Cosine Similarity**

This project also includes a clean and interactive **Streamlit Web Application** that enables real-time customer segment prediction and product recommendations.

---

## Key Objectives
- Perform complete **EDA** to understand sales and customer behavior patterns
- Build **RFM features** (Recency, Frequency, Monetary) for customer profiling
- Apply **clustering models** and compare them to select the most reliable segmentation method
- Build an **item-based recommendation engine** to recommend similar products
- Deploy everything through a **Streamlit multi-page UI** with strong UX

---

## Dataset Details
**File Name:** `online_retail.csv`  
**Rows:** 541,909  
**Main Columns:**
- `InvoiceNo`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `UnitPrice`
- `CustomerID`
- `Country`

---

## Tech Stack
**Python | Pandas | NumPy | Scikit-learn | Matplotlib | Seaborn | SciPy | Joblib | Streamlit**  

---

## Project Workflow

### 1. Data Loading & Validation
- Loaded full dataset without skipping rows/columns
- Verified columns, shape, and sample records

Script: `load_check.py`

---

### 2. Data Cleaning & Feature Engineering
Cleaning steps applied:
- Removed missing `CustomerID`
- Removed cancelled invoices (`InvoiceNo` starting with `"C"`)
- Removed invalid entries (`Quantity <= 0`, `UnitPrice <= 0`)
- Created new feature: `TotalPrice = Quantity * UnitPrice`
- Saved optimized dataset to Parquet format

Script: `clean_data.py`  
Output: `data/online_retail_cleaned.parquet`

---

### 3. Exploratory Data Analysis (EDA)
Performed insights-driven EDA including:
- Top countries by transaction volume
- Top selling products by quantity
- Daily sales trends
- Monetary distribution per transaction and per customer
- RFM feature distributions

Scripts:
- `eda_analysis.py`
- `eda_monetary.py`

Key outputs saved in `data/` folder as `.png` charts.

---

### 4. RFM Analysis (Customer Profiling)
Built RFM features per customer:
- **Recency:** Days since last purchase
- **Frequency:** Number of unique invoices
- **Monetary:** Total spend per customer

Script: `rfm_build.py`  
Output: `data/rfm_table.csv`

---

### 5. Customer Segmentation (Clustering)
#### Step 1: Normalize RFM
Used `StandardScaler` to standardize RFM values before clustering.

#### Step 2: Cluster Selection
Used:
- **Elbow Method** (inertia curve)
- **Silhouette Score** (best k selection)

Scripts:
- `clustering_elbow.py`
- `find_best_k.py`

#### Step 3: Run Clustering Models
We evaluated and compared **three clustering algorithms**:
- **KMeans (k=4)**
- **Agglomerative Clustering**
- **DBSCAN**

Model comparison metrics used:
- Silhouette Score
- Davies–Bouldin Index
- Calinski–Harabasz Score

Script: `kmeans_clustering.py`  
Comparison Script: `compare_clustering_models.py` *(if included)*  
Visualization Script: `advanced_model_comparison_viz.py`

Final choice: **KMeans (k=4)**  
Reason: Best balance of interpretability + actionable segmentation.

---

### 6. Segment Labeling (Business Interpretation)
Final segment labels were mapped based on RFM behavior:
- **High Value**
- **Regular**
- **Occasional**
- **At Risk**

Script: `segment_labeling.py`  
Output: `data/customer_segments_labeled.csv`

---

### 7. Product Recommendation System (Item-Based Collaborative Filtering)
Implemented item-to-item recommendations using:
- CustomerID–Description purchase matrix
- Cosine similarity between product vectors
- Top 5 product recommendations

Script:
- `recommendation_model.py`

Saved for Streamlit:
- Similarity matrix
- Product list

Script:
- `save_recommendation_data.py`

Outputs:
- `models/product_similarity.pkl`
- `models/product_list.pkl`

---

## Hypothesis Testing (Advanced Validation)
Three statistical hypothesis tests were performed using **SciPy**:

### 1) Country vs Revenue Difference (UK vs Germany)
- Test: Mann–Whitney U Test
- Result: Significant difference found

### 2) Cluster Spending Difference (Monetary across clusters)
- Test: Kruskal–Wallis Test
- Result: Significant spending differences across segments

### 3) Recommendation Quality vs Random Baseline
- Test: Mann–Whitney U Test (one-sided)
- Result: Recommended products have significantly higher cosine similarity than random products

Scripts:
- `hypothesis_test_1_country_revenue.py`
- `hypothesis_test_2_cluster_monetary.py`
- `hypothesis_test_3_recommendation_quality.py`

---

## Streamlit Web Application
A multi-page Streamlit app was built with clean UI and navigation.

### Pages Included
1. **Customer Segmentation**
   - Input: Recency, Frequency, Monetary
   - Output: Predicted cluster + segment label + recommended action

2. **Product Recommendation**
   - Select a product
   - Output: Top 5 similar products using cosine similarity

3. **Business Insights Dashboard**
   - Key KPIs: Customers, Transactions, Revenue, Unique Products
   - Segment distribution table
   - Real-world business use cases

Files:
- `app.py`
- `pages/1_Customer_Segmentation.py`
- `pages/2_Product_Recommendation.py`
- `pages/3_Business_Insights.py`

---

## Real-time Business Use Cases Delivered
- Customer Segmentation for Targeted Marketing Campaigns
- Personalized Product Recommendations for E-Commerce Platforms
- Identifying At-Risk Customers for Retention Programs
- Dynamic Pricing Strategies Based on Purchase Behavior
- Inventory Management & Stock Optimization using demand patterns

---
##FILES

ShopperSpectrum/
│── app.py
│── clean_data.py
│── load_check.py
│── rfm_build.py
│── clustering_elbow.py
│── kmeans_clustering.py
│── segment_labeling.py
│── recommendation_model.py
│── save_recommendation_data.py
│── eda_analysis.py
│── eda_monetary.py
│
├── data/
│   ├── online_retail.csv
│   ├── online_retail_cleaned.parquet
│   ├── rfm_table.csv
│   ├── customer_segments.csv
│   ├── cluster_summary.csv
│   ├── customer_segments_labeled.csv
│   ├── *.png
│
├── models/
│   ├── scaler.pkl
│   ├── kmeans_model_k4.pkl
│   ├── segment_map.json
│   ├── product_similarity.pkl
│   ├── product_list.pkl
│
└── pages/
    ├── 1_Customer_Segmentation.py
    ├── 2_Product_Recommendation.py
    ├── 3_Business_Insights.py

```bash
python -m venv venv


##Conclusion

Shopper Spectrum successfully combines analytics + machine learning + business intelligence into a deployable solution.
This project demonstrates practical skills in:
Large dataset handling
Feature engineering (RFM)
Clustering model comparison
Recommendation systems
Statistical hypothesis testing
Streamlit deployment and dashboarding
