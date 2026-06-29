# Bengaluru House Price Predictor

A machine learning project that predicts house prices in Bengaluru using property details like area, BHK, bathrooms, and location.
The project compares three different modeling approaches and explains why the highest-scoring model isn't always the one that got deployed.

**Deployed Project:**  https://bengaluru-house-price-predictor-moln.onrender.com

## Overview

Most house price prediction projects stop at reporting an R² score.
This project goes a step further by testing how each model behaves on individual predictions, not just on the aggregate metric,
and uses that to decide which model actually gets deployed.

Three models were trained and compared: 

 Linear Regression : R² Score : 0.855 
 XGBoost : R² Score : 0.859 
 Neural Network : R² Score : 0.899 

On paper, the Neural Network looks like the clear winner.
But testing it on individual examples (same area, BHK, and bathrooms, different location) revealed a problem:
predictions barely changed across different locations, even though location is one of the strongest price drivers in real estate.
XGBoost showed a milder version of the same issue.

Investigating further showed the dataset was imbalanced across locations, some areas had many listings, most had very few.
With limited examples per location, the models leaned more heavily on features like area and BHK, which were easier to learn from.
The Neural Network's higher R² wasn't capturing a better relationship, it was fitting patterns and noise in an imbalanced dataset more aggressively.

**XGBoost was deployed** as the final model.
It has a slightly lower R² than the Neural Network, but more stable behavior across edge cases and is easier to interpret.

## Key takeaway

A higher evaluation metric doesn't automatically mean a better real-world model.
Understanding the dataset, its imbalances and limitations, often matters more than picking the most complex algorithm.

## Tech stack

- Python
- pandas, NumPy (data processing)
- scikit-learn (Linear Regression, preprocessing)
- XGBoost (deployed model)
- Streamlit (web app)

## Project structure

```
bengaluru-house-price-predictor/
├── data/                  # Raw and cleaned dataset
├── notebooks/             # EDA and model experimentation
├── models/                # Saved trained model (XGBoost)
├── app.py                 # Streamlit application
├── train.py                # Model training script
├── requirements.txt
└── README.md
```

## How it works

1. **Data cleaning** — handling missing values, removing outliers, standardizing location names
2. **Feature engineering** — converting categorical features (location) into a usable format, deriving price-per-sqft
3. **Model training** — training Linear Regression, XGBoost, and a Neural Network on the same processed data
4. **Evaluation** — comparing R² scores, then testing individual predictions to check for hidden inconsistencies
5. **Deployment** — packaging the XGBoost model into a Streamlit app for interactive predictions

## Dataset

The dataset used is the Bengaluru House Price dataset, containing property listings with features such as area type, location,
size (BHK), total square footage, number of bathrooms, and price.


## Future improvements

- Address location data imbalance with better sampling or grouping of rare locations
- Add confidence intervals to predictions instead of point estimates
- Expand feature set (e.g. proximity to amenities, age of property)
- Retrain Neural Network with balanced location representation to see if the inconsistency resolves

## Author
Sawan Kushwah
