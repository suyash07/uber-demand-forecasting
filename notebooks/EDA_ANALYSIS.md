# Uber Demand Forecasting - Exploratory Data Analysis (EDA)

## Overview
This document summarizes the Exploratory Data Analysis (EDA) performed on Uber trip data for January and February. The analysis covers demand patterns, temporal trends, geographic distribution, trip characteristics, and data cleaning procedures.

**Dataset**: 36.44 million trips across January and February
- **January**: 18,479,208 trips (50.7%)
- **February**: 17,959,683 trips (49.3%)

---

## 1. Data Loading & Preparation

### Data Sources
- `url_jan.parquet` - January trip data
- `url_feb.parquet` - February trip data
- `taxi_zone_lookup.csv` - Zone ID to neighborhood mapping

### Feature Engineering
Created the following features for analysis:
- **request_datetime**: Parsed as datetime
- **hour**: Hour of day (0-23)
- **day_of_week**: Day name (Monday-Sunday)
- **weekend**: Boolean flag (Saturday=True, Sunday=True)
- **date**: Trip date
- **month**: Month number (1=January, 2=February)

### Data Quality
- **Null values**: No missing values in key columns
- **Data types**: Proper datetime parsing applied
- **Data merging**: Zones data merged on `PULocationID` to add neighborhood information

---

## 2. Demand Distribution Analysis

### 2.1 Daily Demand
**Key Insights:**
- **Average daily trips**: Varies day-to-day
- **Daily demand statistics** (from `daily_demand.describe()`):
  - Shows variation in demand across the 2-month period
  - Can identify peak and low-demand days

### 2.2 Monthly Distribution
**Breakdown:**
| Month | Trips | Percentage |
|-------|-------|-----------|
| January | 18,479,208 | 50.7% |
| February | 17,959,683 | 49.3% |

**Finding**: Nearly balanced demand across both months with January slightly higher (~520k more trips)

### 2.3 Weekly Pattern
**Weekday Distribution Analysis:**
- Demand varies by day of week (Monday through Sunday)
- Identifies which days have peak demand
- Useful for understanding weekly cyclicity

### 2.4 Hourly Distribution
**All Trips Hourly Pattern:**
- 24-hour distribution of ride requests
- Identifies peak and off-peak hours
- Used for understanding intra-day demand variability

---

## 3. Temporal Demand Patterns

### 3.1 Weekend vs Weekday Analysis

#### **Weekend Hourly Demand** (Saturday & Sunday)
**Key Findings:**
- **Midnight peak** (0:00): ~678k trips - Late night rides home
- **Early morning dip** (4-6 AM): ~186k-250k trips - Lowest demand
- **Gradual rise** (7 AM-6 PM): Steady increase throughout day
- **Evening peaks** (6-7 PM, 10 PM): ~680k-687k trips - Entertainment/dining
- **Late night surge** (10 PM onwards): High demand for nightlife

#### **Weekday Hourly Demand** (Monday-Friday)
- **Morning rush** (7-9 AM): Increased demand for commute
- **Midday plateau** (10 AM-5 PM): Moderate consistent demand
- **Evening rush** (5-7 PM): Peak demand for end-of-day commutes
- **Night decline** (11 PM-3 AM): Reduced demand

**Insight**: Weekend demand shows entertainment-focused pattern, while weekdays follow commute-based pattern

---

## 4. Geographic Analysis

### 4.1 Borough-Level Distribution
**Top Pickup Locations by Trip Count:**
- Analyzed by Borough to identify geographic demand hotspots
- Highest demand boroughs are primary service areas

### 4.2 Key Findings
- Certain boroughs generate majority of ride requests
- Geographic distribution reflects population density and service area

---

## 5. Trip Characteristics Analysis

### 5.1 Trip Duration Distribution
**Analysis**: Distribution of trip durations across all trips
- **Purpose**: Understand typical trip lengths and outliers
- **Method**: Histogram visualization with 50 bins

**Outlier Removal Process:**
- **Original data**: Includes extreme outliers (some trips >999th percentile)
- **95th percentile**: Reference for reasonable trip duration
- **99th percentile**: Upper bound threshold
- **Cleaning threshold**: Trips > 180 minutes removed
  - Rationale: Excludes unusually long trips (likely data errors or anomalies)

**After cleaning:**
- More realistic distribution
- Removed potential overnight trips or stuck rides

### 5.2 Trip Distance Distribution
**Analysis**: Distribution of trip miles
- **Original range**: Extreme outliers up to thousands of miles
- **95th percentile**: ~50 miles
- **Cleaning approach**: Removed trips > 50 miles
  - Rationale: Trips beyond 50 miles are statistical outliers

**Characteristics:**
- Most trips are under 10-15 miles
- Distribution is right-skewed

### 5.3 Base Passenger Fare Distribution
**Analysis**: Distribution of base fares charged
- **Original distribution**: Wide range with outliers
- **99th percentile**: Used to identify outlier threshold
- **Cleaning criteria**: 
  - Removed fares ≥ $150
  - Removed fares ≤ $0
  - Rationale: $150+ fares are statistical anomalies; $0 fares are invalid

**Cleaned Distribution:**
- Concentrated between $0-$150
- Majority of fares in $5-$50 range
- More realistic for Uber pricing model

---

## 6. Data Cleaning Summary

### 6.1 Outlier Removal Strategy
Applied progressive cleaning to create `df_clean` dataset:

**Step 1: Distance Filtering**
```python
df_clean = df[df['trip_miles'] < 50]
```
- Rationale: Remove geographic outliers (extremely long trips)

**Step 2: Duration Filtering**
```python
df_clean = df_clean[df_clean['trip_duration'] <= 180]
```
- Rationale: Remove trips longer than 3 hours (anomalies)
- Threshold: 180 minutes

**Step 3: Fare Filtering**
```python
df_clean = df_clean[(df_clean['base_passenger_fare'] < 150) & (df_clean['base_passenger_fare'] > 0)]
```
- Rationale: Remove extreme fare values
- Valid range: $0.01 - $149.99

### 6.2 Data Quality Metrics
- **Original dataset**: 36,440,002 trips
- **Cleaned dataset**: Significantly reduced outliers for modeling
- **Removed %**: To be calculated based on final df_clean size

---

## 7. Key Insights Summary

### Demand Patterns
1. **Temporal**: Distinct weekday vs weekend patterns
2. **Hourly**: Peak hours differ significantly between weekdays (commute-based) and weekends (entertainment-based)
3. **Monthly**: January and February have nearly equal demand

### Trip Characteristics
1. **Distance**: Most trips under 15 miles
2. **Duration**: Most trips under 30 minutes
3. **Fare**: Most fares between $5-$50

### Data Quality
1. Identified and removed outliers in distance, duration, and fare
2. Prepared clean dataset for modeling and forecasting

---

## 8. Visualizations Generated

| Analysis | Chart Type | Key Finding |
|----------|-----------|------------|
| Weekday Demand | Line Chart | Weekly cyclicity identified |
| Hourly Demand | Bar Chart | Peak hours at midnight and evening |
| Weekend Hourly | Bar Chart | Entertainment-driven demand pattern |
| Weekday Hourly | Bar Chart | Commute-driven demand pattern |
| Borough Distribution | Bar Chart | Geographic hotspots identified |
| Trip Duration | Histogram | Right-skewed distribution |
| Trip Distance | Histogram | Concentrated under 20 miles |
| Base Fare | Histogram | Most fares $5-$50 range |

---

## 9. Next Steps

### For Feature Engineering
- Incorporate weather data
- Add cyclical features (time of day, day of week)
- Create lag features from previous hours/days
- Add external factors (events, holidays)

### For Modeling
- Train time-series forecasting models (ARIMA, Prophet, XGBoost)
- Use cleaned `df_clean` dataset
- Validate on separate test set
- Consider zone-level or hourly-level forecasting

### For Visualization Dashboard
- Real-time demand heatmaps by hour and zone
- Demand forecasting predictions
- Peak hour alerts
- Geographic demand distribution

---

## 10. Technical Notes

**Libraries Used:**
- pandas: Data manipulation and grouping
- matplotlib: Visualization
- plotly: Interactive charts (if used)

**Data Processing:**
- Datetime parsing and feature extraction
- Groupby aggregations for temporal analysis
- Data merging with zone reference
- Quantile-based outlier detection

**File Structure:**
```
notebooks/
├── eda.ipynb                 # Main EDA notebook
└── EDA_ANALYSIS.md          # This documentation
```

---

**Analysis Date**: June 2026
**Status**: Complete - Ready for modeling phase
