# Uber Demand Forecasting

Machine learning project for analyzing and forecasting Uber demand patterns using 36.44 million trips from January and February.

## Project Overview
This project combines exploratory data analysis, data cleaning, feature engineering, and machine learning to build a demand forecasting system for Uber rides.

**Dataset Size**: 36.44 million trips (January + February)

## Contents

### 📊 Exploratory Data Analysis
See [EDA_ANALYSIS.md](notebooks/EDA_ANALYSIS.md) for detailed analysis covering:
- Demand distribution (daily, hourly, weekly, monthly)
- Temporal patterns (weekday vs weekend)
- Geographic analysis (pickup locations)
- Trip characteristics (duration, distance, fare)
- Data cleaning and outlier removal

**Key Findings**:
- January: 18.48M trips | February: 17.96M trips
- Weekend demand driven by entertainment (peak at midnight, 10 PM)
- Weekday demand driven by commutes (peak at 8 AM, 5-6 PM)
- Most trips under 15 miles and 30 minutes
- Cleaned dataset removing distance/duration/fare outliers

## 🔍 Key Findings

### Temporal Demand Patterns
- **Weekday vs Weekend**: Discovered distinct usage patterns driven by commutes (weekdays) vs entertainment (weekends)
  - **Weekdays**: Morning peak at 7-9 AM, evening peak at 5-7 PM
  - **Weekends**: Midnight peak (678k trips), evening peaks at 6-7 PM & 10 PM
- **Hourly Analysis**: 24-hour demand distribution reveals 4 distinct peaks for demand optimization

### Data Quality & Cleaning
- Processed **36.44M trips** with rigorous outlier detection
- **Distance**: Removed trips >50 miles (geographic outliers)
- **Duration**: Removed trips >180 minutes (3 hours - likely data errors)
- **Fare**: Removed fares ≥$150 and ≤$0 (invalid pricing data)
- Methodology: Used quantile analysis (95th, 99th, 99.9th percentiles) to identify thresholds

### Geographic Distribution
- Analyzed pickup location patterns by borough
- Identified high-demand areas for service optimization
- Geographic concentration reveals potential for targeted resource allocation

### Dataset Overview
- **Total Trips**: 36.44 million
- **January**: 18.48M trips (50.7%)
- **February**: 17.96M trips (49.3%)
- **Data Quality**: Identified and documented data anomalies (March, December outliers)

## 📁 Directory Structure
```
├── data/                   # Raw parquet and CSV files
│   ├── url_jan.parquet    # January trip data
│   ├── url_feb.parquet    # February trip data
│   └── taxi_zone_lookup.csv
├── notebooks/             # Analysis and exploration
│   ├── eda.ipynb          # Full EDA notebook
│   └── EDA_ANALYSIS.md    # EDA documentation
├── src/                   # Reusable modules
│   ├── features.py        # Feature engineering
│   ├── model.py           # Model training/evaluation
│   └── utils.py           # Utility functions
├── dashboard/             # Dash visualization app
│   └── app.py
└── requirements.txt       # Dependencies
```

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Run EDA Notebook
```bash
jupyter notebook notebooks/eda.ipynb
```

## 🛠️ Skills Demonstrated

- **Time-series Analysis**: Identified temporal patterns and demand cycles
- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of 36M+ records
- **Data Cleaning & Preprocessing**: Quantile-based outlier detection and removal
- **Statistical Analysis**: Distribution analysis, percentile calculations
- **Geospatial Analysis**: Geographic demand distribution by borough
- **Data Visualization**: Multiple chart types for pattern discovery
- **Business Insight Translation**: Converting data patterns into actionable insights
- **Documentation**: Clear communication of findings and methodology

## Next Steps
- Feature engineering with weather and external data
- Time-series forecasting models (ARIMA, Prophet, XGBoost)
- Build interactive dashboard for predictions
- Deploy forecasting API
