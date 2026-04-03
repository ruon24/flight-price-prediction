# Flight Price Prediction

A machine learning project to predict flight prices using regression models with extensive feature engineering and model comparison.

##  Project Overview

This project analyzes and predicts flight prices based on various features such as airline, route, departure time, booking window, and flight characteristics. The project demonstrates a complete end-to-end machine learning workflow including data quality assessment, feature engineering, model training, and comprehensive model comparison.

##  Objectives

* Build accurate regression models to predict flight prices
* Identify key features that influence flight pricing
* Compare multiple machine learning algorithms to find the best performer
* Engineer meaningful features from raw data to improve model performance

## 📁 Dataset

**Source**: `/Volumes/workspace/default/mldatabases/Clean_Dataset.csv`

**Features**:
* `airline`: Airline name (6 unique airlines)
* `flight`: Flight identifier
* `source_city`: Departure city (6 cities)
* `departure_time`: Time period of departure (Early_Morning, Morning, Afternoon, Evening, Night, Late_Night)
* `stops`: Number of stops (zero, one, two_or_more)
* `arrival_time`: Time period of arrival
* `destination_city`: Arrival city (6 cities)
* `class`: Travel class (Business, Economy)
* `duration`: Flight duration in hours
* `days_left`: Days until departure
* `price`: Target variable (flight price in ₹)

**Data Quality**:
* ✓ No missing values
* ✓ No duplicate rows
* Clean dataset ready for analysis

## 🔍 Exploratory Data Analysis

### Key Insights:

1. **Price Distribution**:
   * Significant price variation across different routes and airlines
   * Outliers present but represent legitimate high-value tickets

2. **Categorical Features**:
   * 6 airlines with varying price ranges
   * 36 unique routes (6 cities × 6 destinations)
   * Class distribution: 68.85% Economy, 31.15% Business
   * Most flights are direct (zero stops)

3. **Numerical Features**:
   * Duration, days_left, and price show distinct distributions
   * Strong correlations identified between engineered features and price

## 🛠️ Feature Engineering

Extensive feature engineering was performed to extract maximum predictive power:

### 1. **Route Features**
* `route`: Concatenation of source and destination cities
* `route_encoded`: Mean price encoding for each route
* `route_popularity`: Frequency of each route
* `route_days`: Interaction between route encoding and booking window

### 2. **Temporal Features**
* `last_minute`: Binary flag for bookings ≤3 days before departure
* `booking_window`: Categorized booking periods (ultra_last, last, short, medium, early)
* `days_left_log`: Log transformation of days until departure
* `dep_hour`/`arr_hour`: Mapped time periods to representative hours
* `dep_period`: Re-categorized departure hours into periods
* `dep_sin`/`dep_cos`: Cyclical encoding of departure hour

### 3. **Flight Characteristics**
* `is_direct`: Binary flag for direct flights
* `is_multi_stop`: Binary flag for multi-stop flights
* `stops_factorized`: Numerical encoding of stop categories
* `duration_log`: Log transformation of flight duration
* `long_flight`: Binary flag for flights longer than median
* `duration_stops`: Interaction between duration and multi-stop flights

### 4. **Airline Features**
* `airline_encoded`: Mean price encoding for each airline
* `airline_days`: Interaction between airline encoding and booking window

### 5. **Class Encoding**
* Binary encoding: Business = 1, Economy = 0

### Feature Selection
After engineering, non-numeric categorical columns were dropped, retaining 20 numerical features for modeling.

## 🤖 Models & Results

### Models Evaluated:

1. **Baseline (Mean Predictor)**
2. **Linear Regression**
3. **Decision Tree**
4. **Gradient Boosting**
5. **Random Forest**
6. **XGBoost**
7. **LightGBM**
8. **CatBoost**

### Performance Comparison:

| Model | R² Score | MSE | MAE (₹) |
|-------|----------|-----|----------|
| **Random Forest** | **0.9852** | **7,616,365** | **1,099.16** |
| Decision Tree | 0.9756 | 12,602,901 | 1,195.83 |
| XGBoost | 0.9752 | 12,806,372 | 2,000.17 |
| CatBoost | 0.9747 | 13,059,052 | 2,031.87 |
| LightGBM | 0.9705 | 15,183,622 | 2,251.38 |
| Gradient Boosting | 0.9561 | 22,628,910 | 2,755.44 |
| Linear Regression | 0.9092 | 46,807,293 | 4,470.57 |
| Baseline (Mean) | -0.0000 | 515,482,303 | 19,768.69 |

### 🏆 Best Model: Random Forest

* **R² Score**: 0.9852
* **MAE**: ₹1,099.16
* **Improvement over Baseline**: 94.4% MAE reduction

**Why Random Forest Performed Best:**
* Effectively captures non-linear relationships
* Handles interactions between features well
* Robust to outliers
* Minimal overfitting with default parameters

## 🚀 Getting Started

### Prerequisites

```python
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
catboost
```

### Installation

```bash
%pip install xgboost lightgbm catboost
```

### Running the Notebook

1. **Load Data**: Execute cells 1-2 to load the dataset
2. **Data Quality Assessment**: Run cells 3-11 for EDA
3. **Feature Engineering**: Execute cells 12-24 to create features
4. **Preprocessing**: Run cells 25-28 for encoding and train/test split
5. **Model Training**: Execute cells 30-36 for baseline and comprehensive model comparison
6. **Hyperparameter Tuning** (Optional): Run cell 38 for RandomizedSearchCV on Random Forest

## 📈 Project Structure

```
flight-price-prediction/
│
├── flight price prediction.ipynb    # Main notebook with complete workflow
├── README.md                          # Project documentation
│
└── Data/
    └── Clean_Dataset.csv              # Clean flight data (300K+ rows)
```

## 📊 Key Workflow Steps

1. **Data Loading & Inspection**
   * Load dataset and display basic information
   * Check data types and structure

2. **Data Quality Assessment**
   * Missing values check
   * Duplicate detection
   * Statistical summaries
   * Outlier analysis
   * Distribution visualization

3. **Feature Engineering**
   * Create temporal features
   * Encode categorical variables
   * Generate interaction features
   * Apply transformations (log, cyclical encoding)

4. **Data Preprocessing**
   * Drop redundant columns
   * Select numerical features only
   * Train/test split (80/20)

5. **Model Training**
   * Train baseline model
   * Train 7 regression models
   * Compare performance metrics

6. **Model Evaluation**
   * R² Score
   * Mean Squared Error (MSE)
   * Mean Absolute Error (MAE)

7. **Hyperparameter Tuning** (Optional)
   * RandomizedSearchCV for best model
   * Cross-validation

## 💡 Key Findings

1. **Feature Importance**:
   * Class (Business vs Economy) is the strongest predictor
   * Route characteristics heavily influence pricing
   * Booking window has significant impact
   * Flight duration and number of stops are important factors

2. **Model Performance**:
   * Tree-based models significantly outperform linear models
   * Random Forest achieves best balance of accuracy and generalization
   * Ensemble methods consistently perform well

3. **Business Insights**:
   * Business class tickets command ~3x premium
   * Last-minute bookings are significantly more expensive
   * Direct flights typically cost more than flights with stops
   * Certain routes have consistently higher prices (popular/long-distance)

## 🔧 Hyperparameter Tuning

For further optimization, RandomizedSearchCV is configured for Random Forest:

**Parameter Grid**:
* `n_estimators`: [100, 200, 300]
* `max_depth`: [10, 20, 30, None]
* `min_samples_split`: [2, 5, 10]
* `min_samples_leaf`: [1, 2, 4]

**Configuration**:
* 20 random parameter combinations
* 3-fold cross-validation
* Scoring: Negative MAE

## 📝 Future Improvements

* [ ] Collect more data for seasonal patterns
* [ ] Add external features (fuel prices, holidays, events)
* [ ] Implement deep learning models (Neural Networks)
* [ ] Create a deployment-ready prediction API
* [ ] Add feature importance visualization
* [ ] Perform ablation study to identify critical features
* [ ] Experiment with ensemble stacking

## 👨‍💻 Author

**Nour Bichiou**
* Email: nour.bichiou@insat.ucar.tn

## 📄 License

This project is available for educational and research purposes.

## 🙏 Acknowledgments

* Dataset source: Clean_Dataset.csv
* Built using Databricks platform
* Leverages scikit-learn, XGBoost, LightGBM, and CatBoost libraries

---

**Last Updated**: April 2026
