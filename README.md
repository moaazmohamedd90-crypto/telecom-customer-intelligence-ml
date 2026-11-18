# 📊 Telecom Customer Intelligence — Machine Learning Capstone

An end-to-end machine learning and data science project designed to analyze customer behavior, predict churn, estimate monthly billing, discover customer segments, and recommend telecom service bundles.

## Project Overview

Nile Connect is a fictional telecommunications company operating in Egypt.

This project explores how machine learning can help telecom businesses understand customer behavior and make data-driven decisions.

The analysis covers supervised learning, unsupervised learning, dimensionality reduction, and association rule mining.

## Business Objectives

1. Predict customers at risk of churn.
2. Estimate monthly customer bills.
3. Identify customer segments using behavioral data.
4. Discover frequently purchased service combinations.
5. Translate machine learning findings into business recommendations.

## Dataset

The project uses three telecom datasets:

| Dataset | Description |
|---|---|
| customers.csv | Customer demographics, subscriptions, billing, and churn |
| usage_behavior.csv | Customer usage patterns for clustering |
| service_transactions.csv | Telecom service transactions for association rules |

The datasets are used for educational and portfolio purposes.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- Matplotlib and Seaborn
- MLxtend
- Jupyter Notebook
- Joblib (optional model persistence)

## Machine Learning Workflow

### 1. Data Cleaning and Exploratory Data Analysis

- Missing-value analysis
- Duplicate detection and removal
- Outlier analysis
- Categorical data cleaning
- Exploratory visualizations

### 2. Monthly Billing Prediction

Implemented Simple Linear Regression and Multiple Linear Regression.

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Simple Linear Regression | 0.5002 | 104.50 | 131.56 |
| Multiple Linear Regression | 0.9317 | 36.38 | 48.63 |

**Best model:** Multiple Linear Regression.

### 3. Customer Churn Prediction

Evaluated seven machine learning algorithms:

- Logistic Regression
- Decision Tree
- Support Vector Machine
- K-Nearest Neighbors
- Naive Bayes
- Random Forest
- XGBoost

**Selected Model: Logistic Regression**

| Metric | Test Score |
|---|---:|
| Accuracy | 85.17% |
| Precision | 74.81% |
| Recall | 64.74% |
| F1-Score | 69.42% |

### 4. Cross-Validation and Hyperparameter Tuning

Applied:

- Stratified K-Fold Cross-Validation
- GridSearchCV
- RandomizedSearchCV
- Model comparison using F1-Score

### 5. Customer Segmentation

Applied KMeans, Hierarchical Clustering, DBSCAN, and PCA to behavioral data.

**Selected segmentation:** KMeans with 5 clusters.

Identified segments:

1. Digital Entertainment Users
2. Light Usage Customers
3. Social Media Users
4. International & Voice Users
5. Extreme Night & Data Users

KMeans achieved a Silhouette Score of 0.5534 on the standardized original features.

PCA reduced the feature space from 12 dimensions to 5 while preserving approximately 92.42% of the variance.

### 6. Association Rule Mining

Applied the Apriori algorithm to identify frequent service combinations.

**Recommended Bundles:**

- Home Internet + Router Upgrade
- International Calling + Roaming Pack
- Device Installment + Antivirus

### 7. Churn and Segmentation Integration

Investigated whether KMeans cluster labels improved churn classification.

| Model | Mean CV F1 |
|---|---:|
| Logistic Regression Baseline | 0.7334 |
| Logistic Regression + Cluster | 0.7246 |

Adding cluster information did not improve average cross-validation performance in this experiment.

### 8. Executive Business Recommendations

The analysis recommends:

- Using churn predictions to support retention campaigns.
- Applying regression for billing estimation.
- Personalizing marketing offers through customer segmentation.
- Testing targeted service bundles.
- Monitoring model performance before operational deployment.

## How to Run

1. Clone or download the repository.
2. Install Python and Jupyter Notebook.
3. Install the project dependencies.
4. Place the datasets in the `DATA/` folder.
5. Open the project notebook.
6. Select the correct Python kernel.
7. Run all notebook cells from top to bottom.

## Model Evaluation

The project uses appropriate evaluation metrics for each learning task:

- Regression: R², MAE, RMSE
- Classification: Accuracy, Precision, Recall, F1
- Clustering: Silhouette Score and Inertia
- Association Rules: Support, Confidence, Lift

## Limitations

- This is an educational capstone using simulated data.
- The reported model results are experimental and do not establish production readiness.
- Association rules show relationships, not causation.
- Commercial recommendations require validation through real-world testing.

## Project Status

Completed machine learning capstone, including data analysis, model evaluation, customer segmentation, business recommendations, and executive reporting.

## Author

[Your Name]

**Focus:** Data Science | Machine Learning | Python
