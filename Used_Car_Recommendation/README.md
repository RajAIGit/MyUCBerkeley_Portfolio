# Used Car Price Analysis

## Problem Statement
The provided dataset contains information on 426K cars. Your goal is to understand what factors make a car more or less expensive. As a result of your analysis, you should provide clear recommendations to your client -- a used car dealership -- as to what consumers value in a used car.

## Approach
1. Load and clean the dataset.
2. Perform exploratory data analysis (EDA) to understand price distribution and key trends.
3. Use machine learning (Random Forest) to identify feature importance.
4. Provide actionable recommendations based on findings.

## How to Run
1. Ensure you have Python 3 and Jupyter Notebook installed.
2. Install required libraries: pandas, numpy, plotly, scikit-learn.
3. Open `used_car_analysis.ipynb` in Jupyter Notebook and run all cells.

## Deliverables
- Jupyter Notebook: `Used_car_Recommendation.ipynb`
- README file: `README.md`


##Summary of Findings

Summary Report: What Drives the Price of a Car?

Business Objective
To identify key factors influencing used car prices and provide actionable recommendations for inventory and pricing strategies.

Key Insights

1. Year and Mileage Are Primary Drivers
Newer cars command higher prices.
Lower odometer readings significantly increase value.

2. Manufacturer Matters
Luxury brands (BMW, Mercedes, Lexus) have higher average prices compared to economy brands.

3. Condition and Transmission
Cars in good condition and with automatic transmission tend to sell for more.

4. Fuel Type
Gasoline dominates, but hybrids and electric vehicles retain higher prices.


Modeling Results

Linear Regression (Baseline):
R² = 0.13 → Limited explanatory power with only year and mileage.

Enhanced Model with Categorical Features:

R² = 0.42 → Significant improvement by adding manufacturer, fuel, transmission, drive, and type.

Ridge & Lasso Regression:

Regularization improved stability and highlighted most impactful features.


Recommendations

1. Inventory Strategy:
Prioritize newer vehicles (2015+) with low mileage.
Include popular and luxury brands for higher margins.

2. Pricing Strategy:
Use predictive models to set competitive prices based on year, mileage, and brand.
Avoid stocking vehicles with extremely high mileage or very old models unless priced aggressively.

3. Expand Hybrid/Electric Options:
These vehicles show higher price retention and appeal to eco-conscious buyers.

4. Data-Driven Decisions:
Regularly update pricing models with new data.
Consider adding more features (e.g., color, size, drive type) for better prediction accuracy.

Next Steps

1. Deploy predictive pricing tool for dealership.
2. Train staff on data-driven inventory selection.
3. Explore partnerships for sourcing hybrid/electric vehicles.

Note: The file vehicles.csv used for this analysis was over 40MB hence was not uploaded to GIT
