# Bank Term Deposit Subscription Prediction

## Project Overview
This project aims to predict whether a bank client will subscribe to a term deposit as part of a marketing campaign. The business goal is to increase the efficiency of directed campaigns by reducing unnecessary contacts while maximizing successful subscriptions. 

We focus on using customer demographic and financial information to build machine learning models that classify whether a customer is likely to subscribe.

## Business Objective
- Improve marketing efficiency
- Reduce the number of outbound contacts
- Identify the top customers most likely to subscribe
- Frame the problem as a **binary classification** task: predicting `yes` or `no`

## Data Understanding
We used a subset of the Bank Marketing dataset, focusing only on core bank client attributes:
- Age
- Job
- Marital status
- Education level
- Default status
- Housing loan status
- Personal loan status

The target variable `y` indicates whether a client subscribed to the term deposit.

### Handling Missing Values
The dataset includes several `'unknown'` placeholders in categorical fields. These were treated as valid categories rather than missing values.

### Data Types
- Numeric: `age`
- Categorical: job-related and financial status fields
- Target: binary (mapped to 1 for `yes` and 0 for `no`)

## Data Preparation
To prepare the data for modeling:
- Numeric values were imputed using median imputation
- Categorical values were imputed using the most frequent category
- Categorical features were transformed using One-Hot Encoding
- A consistent preprocessing pipeline was used across all models
- The dataset was split into **training (80%)** and **testing (20%)** using stratified sampling

## Baseline Metrics
We established baseline performance using:
- Majority class prediction
- Random guess baselines for ranking metrics

Key baseline metrics include:
- Accuracy (majority classifier)
- PR AUC (equal to prevalence)
- ROC AUC (0.5 for random)
- Precision@k and Lift@k (equal to prevalence and 1.0, respectively)

## First Model: Logistic Regression
A Logistic Regression model was trained using the preprocessed data. This served as our baseline model and established an initial performance benchmark.

## Model Comparison
Several machine learning models were compared using default settings:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Support Vector Machine (SVM)

Each model was evaluated on:
- Training accuracy
- Test accuracy
- Training time

This provided insight into model performance, overfitting tendencies, and computational cost.

## Model Improvement
To improve model performance, we identified several enhancements:
- Hyperparameter tuning (e.g., KNN neighbor count, tree depth, SVM regularization)
- Switching from accuracy to more meaningful metrics, such as:
  - Precision@k
  - Lift@k
  - PR AUC

These metrics better reflect the business goal of targeted outreach.

## Summary of Activities Completed
1. Defined business objective and clarified modeling goal
2. Explored data and identified missing value strategies
3. Prepared features and target with appropriate transformations
4. Conducted train/test split using stratification
5. Built a baseline Logistic Regression model
6. Benchmarked against KNN, Decision Tree, and SVM
7. Identified strategies for hyperparameter tuning and metric adjustment

## Next Steps
- Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- Optimize models based on PR AUC or Precision@k instead of accuracy
- Test with additional features (campaign, call, macroeconomic) while avoiding leakage
- Evaluate business impact of different cutoff strategies

