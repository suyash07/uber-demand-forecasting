
## What we're building — 2 models##

## Model 1 — Demand Forecasting (Regression)##
Predict how many Uber/Lyft rides will be requested in a specific NYC zone in a specific hour.

# "Given it's Tuesday at 8pm in Midtown Manhattan and it's raining — how many rides will be requested in the next hour?"#

- ## Target variable:## `trip_count` (number of trips per hour per zone)
- ## Type:## Regression
- ## Models:## Linear Regression → Random Forest → XGBoost
- ## Evaluate with:## RMSE, MAE, R²

## Model 2 — Surge Pricing Classifier (Classification)##
Predict whether a trip will be surge-priced based on conditions.

> #"Given the hour, zone, and weather — is this trip likely to cost more than usual?"#

- ## Target variable:## `is_surge` (1 = surge, 0 = normal) — defined as `fare_per_mile > 1.5 × median fare_per_mile for that hour`
- ## Type:## Binary Classification
- ## Models:## Logistic Regression → Gradient Boosting Classifier
- ## Evaluate with:## ROC-AUC, F1-score, Precision, Recall

## The features we'll engineer for both models##

| Feature | Why |
|---|---|
| Hour of day (sin/cos) | Peak hours (8am, 5pm) drive demand |
| Day of week (sin/cos) | Weekends vs weekdays behave differently |
| Is weekend / is holiday | Demand spikes on Friday nights, holidays |
| Lag features (1h, 24h, 168h) | Demand 1 hour ago predicts demand now |
| Rolling averages (3h, 24h) | Smooth out noise, capture trends |
| Precipitation, temperature, wind | Rain increases demand significantly |
| Zone ID | Some zones always busier than others |


## The end goal##

Package everything into a ##Python Dash dashboard deployed on GCP## showing:
- Actual vs predicted demand by zone and hour
- Surge zone heatmap
- Feature importance chart
- Model metrics


## Why this project matters for your resume##

It directly mirrors your Zions work but in a consumer tech context:

| Zions Work | Uber Project |
|---|---|
| Loan prepayment forecasting | Ride demand forecasting |
| Macroeconomic lag features | Temporal + weather lag features |
| ±2σ monitoring | Model evaluation metrics |
| Python Dash on GCP | Python Dash on GCP |

Same skills, new domain — exactly what big tech wants to see. You're currently in EDA which is the foundation for all of this. Once EDA is done, feature engineering and modeling will go fast because the patterns are similar to what you already do at Zions.

Ready to continue with the hourly demand chart?