# Telco Customer Churn Prediction

A machine learning project for analyzing customer churn in a telecommunications dataset and predicting whether a customer is likely to leave the service.

## Project Overview

Customer churn is an important problem for telecommunications companies because retaining existing customers can be more cost-effective than acquiring new ones.

This project performs Exploratory Data Analysis (EDA) on the Telco Customer Churn dataset and compares multiple machine learning approaches for predicting customer churn.

The project includes:

- Exploratory Data Analysis
- Churn distribution analysis
- Customer behavior analysis
- Logistic Regression
- Logistic Regression with GridSearchCV
- Random Forest
- Random Forest with RandomizedSearchCV and SMOTE
- Support Vector Machine (SVM)
- Confusion matrices
- Classification reports
- ROC curves
- ROC-AUC evaluation

## Dataset

The project uses the **Telco Customer Churn dataset**.

Dataset size:

- **7,043 customers**
- **21 original features**
- Churn as the target variable

Important features include:

- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

The target variable is:

- `Churn` — whether the customer left the service

## Exploratory Data Analysis

The notebook investigates several aspects of customer churn, including:

- Overall churn distribution
- Monthly charges vs. churn
- Customer tenure distribution
- Payment method distribution
- Churn rate by payment method
- Churn rate by contract type
- Feature correlation using a correlation heatmap

## Machine Learning Models

### 1. Logistic Regression

A Logistic Regression model was trained using:

- StandardScaler
- Logistic Regression
- `class_weight='balanced'`
- Maximum iterations: 5000
- Train/test split: 80/20
- Stratified splitting
- Random state: 42

Results:

| Metric | Score |
|---|---:|
| Accuracy | 74% |
| Precision (Churn) | 51% |
| Recall (Churn) | 79% |
| F1-score (Churn) | 62% |
| ROC-AUC | 0.8414 |

### 2. Logistic Regression with GridSearchCV

GridSearchCV was used to tune the Logistic Regression hyperparameters.

Best parameters:

```text
C = 1
Penalty = l2
Solver = lbfgs
