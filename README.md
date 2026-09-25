# Data-Driven Early Trend Detection and Warning System

A Machine Learning and Data Science project designed to detect emerging fashion trends early using historical Google Trends search volume time-series data and issue early warnings.

---

## 📌 Project Overview
Fashion trends fluctuate rapidly. Waiting for retail sales data often results in late market entry. This project leverages historical Google Trends data (`gtrends.csv`) across **96 fashion variables** (categories, colors, and fabrics) to predict trend velocity **before** peak popularity.

The system calculates **10 time-series input features** using data strictly available up to each time point (eliminating data leakage), trains machine learning models to forecast future trend growth over a 4-week horizon, and delivers actionable early warnings (**NORMAL**, **WATCH**, **ALERT**).

---

## 📂 Project Structure

```
d:\DS PROJECT/
│
├── dataset/
│   ├── gtrends.csv               # Historical search volume (220 weeks x 97 columns)
│   ├── train.csv                 # Fashion product catalog dataset (5,080 rows)
│   ├── test.csv                  # Held-out fashion product test set (497 rows)
│   ├── category_labels.pt        # 32 Category label mappings
│   ├── color_labels.pt           # 10 Color label mappings
│   ├── fabric_labels.pt          # 59 Fabric label mappings
│   └── normalization_scale.npy   # Feature scaling constant (1065.0)
│
├── src/
│   ├── data_loader.py            # Loads raw CSVs, PyTorch label files, & npy scale
│   ├── preprocessing.py          # Data cleaning, missing value handling (-1.0 -> 0)
│   ├── feature_engineering.py    # Computes 10 historical time-series features
│   ├── target_creation.py        # Calculates future-based growth target labels
│   ├── train_models.py           # Time-aware split & model training (LR, RF, XGB)
│   ├── evaluate.py               # Empirical metrics, confusion matrices, & FI
│   ├── warning_engine.py         # Warning status decision engine (NORMAL, WATCH, ALERT)
│   └── product_analysis.py       # Supporting product catalog mapping
│
├── models/                       # Saved trained models & scalers (.joblib)
├── outputs/                      # Evaluation CSVs & confusion matrix charts
├── app.py                        # Streamlit web application
├── requirements.txt              # Python package dependencies
└── README.md                     # Comprehensive project documentation
```

---

## 🔬 Dataset Structure & Analysis

### 1. `gtrends.csv`
- **Rows**: 220 time steps (Weekly data from `2015-10-05` to `2019-12-16`).
- **Columns**: 97 (1 `date` column + 96 fashion trend variables).
- **Attribute Categories**:
  - **Category Labels (32)**: `miniskirt`, `culottes`, `long sleeve`, `tracksuit`, `maxi`, `sheath dress`, etc.
  - **Color Labels (10)**: `yellow`, `brown`, `blue`, `grey`, `green`, `black`, `red`, `white`, `orange`, `violet`.
  - **Fabric Labels (54)**: `acrylic`, `scuba crepe`, `tulle`, `angora`, `faux leather`, `georgette`, `velvet`, `lace`, `cotton`, etc.
- **Data Quality Handling**: Values of `-1.0` (indicating missing/unindexed search volume) are cleaned to `0.0`.

### 2. Supporting Product Catalog (`train.csv` & `test.csv`)
- **Train rows**: 5,080 catalog items.
- **Test rows**: 497 catalog items.
- **Columns**: 12 normalized continuous embeddings (`0`-`11`), season, category, release_date, color, fabric, image_path, extra.
- **Scale Factor**: `normalization_scale.npy` (scalar = `1065.0`).

---

## 🛠️ Feature Engineering & Data Preprocessing Pipeline

### 1. Data Cleaning & Normalization (`outputs/clean_gtrends.csv`)
- **Datetime Parsing**: Converted raw string dates into pandas datetime (`YYYY-MM-DD`) and sorted chronologically.
- **Duplicate & Missing Values**: Verified 0 duplicate dates. Missing/unindexed search metrics encoded as `-1.0` in Google Trends raw data are cleaned to `0.0`. All trend variables are cast to float numeric types.
- **Categorization Metadata (`outputs/trend_metadata.csv`)**: Mapped all 96 trend variables into 3 domain groups using PyTorch label mappings:
  - **Category (27)**: e.g., `miniskirt`, `culottes`, `trapeze dress`
  - **Color (10)**: e.g., `yellow`, `blue`, `red`
  - **Fabric (59)**: e.g., `cotton`, `acrylic`, `tulle`, `georgette`
- **Long-Format Time-Series (`outputs/trend_timeseries.csv`)**: Transformed wide table into long-format time series (`date`, `trend_name`, `trend_type`, `value`).

---

### 2. Time-Series Feature Engineering (11 Features)

For every time step $t$ and trend variable $v$, **11 historical features** are computed using past and current observations ($t' \le t$) **ONLY**:

#### 1. `current_value` ($y_t$)
- **Explanation**: The raw search index at the current week $t$.
- **Formula**: $y_t$

#### 2. `previous_value` ($y_{t-1}$)
- **Explanation**: The search index from the previous week $t-1$.
- **Formula**: $y_{t-1}$ (for $t=0$, $y_0$)

#### 3. `absolute_change` ($\Delta y_t$)
- **Explanation**: The absolute change in search index between week $t$ and week $t-1$.
- **Formula**: $\Delta y_t = y_t - y_{t-1}$

#### 4. `percentage_change` ($\%\Delta y_t$)
- **Explanation**: The relative percentage growth or contraction compared to the previous week.
- **Formula**:
  $$\text{percentage\_change} = \begin{cases} 0.0 & \text{if } y_{t-1} = 0 \text{ and } y_t = 0 \\ \frac{y_t - y_{t-1}}{y_{t-1} + 1e-5} \times 100 & \text{otherwise} \end{cases}$$

#### 5. `moving_average_4` ($MA_4$)
- **Explanation**: Smooths short-term noise by averaging search volume over the past 4 weeks.
- **Formula**: $MA_4(t) = \frac{1}{K} \sum_{i=0}^{K-1} y_{t-i}, \quad \text{where } K = \min(4, t+1)$

#### 6. `rolling_std_4` ($STD_4$)
- **Explanation**: Measures recent variation in search index over the past 4 weeks.
- **Formula**: Sample standard deviation over past $\min(4, t+1)$ observations ($ddof=1$).

#### 7. `moving_average_8` ($MA_8$)
- **Explanation**: Captures medium-term baseline popularity over the past 8 weeks.
- **Formula**: $MA_8(t) = \frac{1}{M} \sum_{i=0}^{M-1} y_{t-i}, \quad \text{where } M = \min(8, t+1)$

#### 8. `trend_slope` ($Slope_4$)
- **Explanation**: Fits an OLS linear regression over the recent 4-week window to measure trend trajectory direction and rate.
- **Formula**:
  $$\text{Slope} = \frac{\sum_{i=0}^{K-1} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=0}^{K-1} (x_i - \bar{x})^2}$$
  - $Slope > 0$: Outbreak / Upward growth trajectory
  - $Slope < 0$: Decline / Contraction
  - $Slope \approx 0$: Stable / Plateau

#### 9. `momentum` ($Mom_t$)
- **Explanation**: Measures acceleration (change in velocity) by taking the difference between current weekly change and previous weekly change.
- **Formula**: $\text{momentum}_t = (y_t - y_{t-1}) - (y_{t-1} - y_{t-2})$ (for $t \ge 2$, else $0.0$)

#### 10. `volatility` ($Vol_t$)
- **Explanation**: Normalized volatility coefficient (coefficient of variation) comparing standard deviation against recent mean popularity.
- **Formula**: $\text{volatility}_t = \frac{\text{rolling\_std\_4}_t}{\text{moving\_average\_4}_t + 1e-5}$

#### 11. `persistence` ($Pers_t$)
- **Explanation**: The number of consecutive weekly periods up to time $t$ during which the trend search index strictly increased without interruption.
- **Formula**: Count of consecutive weeks prior to and including $t$ where $y_k > y_{k-1}$.

---

### 3. Strict Anti-Leakage Guarantee
- **Rule**: At time $t$, features use information $\le t$ **ONLY**.
- Future information ($t+1, t+2, \dots$) is strictly excluded from feature computation and is reserved strictly for constructing target labels during ML training.

---

### 4. Handling Initial Rows
- **Method**: Expanding Window for initial periods ($t < 4$ or $t < 8$).
- **Rationale**: Rather than dropping initial rows or filling with zero, using available historical observations up to $t$ retains full temporal length (220 weeks $\times$ 96 trends = 21,120 rows) without introducing NaN values or artificial zeros.

---

## 🎯 Future-Based Target Creation System

To enable Machine Learning models to predict trend emergence **before** peak popularity occurs, a future-based target variable is constructed over a 4-week lookahead forecast window ($t+1 \dots t+4$).

---

### 1. Rationale & Anti-Leakage Design
- **Why Future Data is Needed for Targets**: Supervised ML models require target labels representing future behavior ($t+1 \dots t+4$) so they can learn patterns in past/current search data ($t' \le t$) that precede trend outbreaks.
- **Strict Anti-Leakage Guarantee**: All future-derived variables (`future_val_t4`, `future_max`, `future_growth_pct`, `next_4_values`) are used **ONLY** to generate the target label (`target`, `target_label`). They are **completely excluded** from the final ML feature set (`outputs/ml_dataset.csv`).

---

### 2. Forecast Horizon & Growth Metrics
For every trend variable at time step $t$, the future window includes observations $[y_{t+1}, y_{t+2}, y_{t+3}, y_{t+4}]$:
- **`future_val_t4`**: Search volume index at $t+4$.
- **`future_max`**: Maximum search volume index in the next 4 weeks ($\max(y_{t+1 \dots t+4})$).
- **`future_mean`**: Average search volume index over the next 4 weeks ($\text{Mean}(y_{t+1 \dots t+4})$).
- **Normalized Growth Formulas**:
  $$\text{Future Growth (\%)} = \frac{y_{t+4} - y_t}{\max(y_t, 5.0)} \times 100$$
  $$\text{Future Max Growth (\%)} = \frac{\max(y_{t+1 \dots t+4}) - y_t}{\max(y_t, 5.0)} \times 100$$
  $$\text{Future Mean Growth (\%)} = \frac{\text{Mean}(y_{t+1 \dots t+4}) - y_t}{\max(y_t, 5.0)} \times 100$$

---

### 3. Data-Driven Threshold Selection & Target Classes

Target classes were selected based on empirical distribution analysis across 20,736 weekly observations:

1. **`Stable` (Class 0)** — **68.80% (14,267 rows)**:
   - Growth remains low ($\text{Future Mean Growth} < 5.0\%$) or fluctuates around baseline.
   - Includes temporary false spikes that collapse back below baseline by week $t+4$.

2. **`Emerging Trend` (Class 1)** — **16.34% (3,388 rows)**:
   - Moderate positive growth ($\text{Future Mean Growth} \ge 5.0\%$ or $\text{Future Max Growth} \ge 15.0\%$).
   - Trend progression is sustained ($\text{Mean}(y_{t+1 \dots t+4}) > y_t$ and $y_{t+4} \ge y_t$).

3. **`Strong Trend` (Class 2)** — **14.86% (3,081 rows)**:
   - Substantial growth surge ($\text{Future Mean Growth} \ge 25.0\%$ or $\text{Future Max Growth} \ge 35.0\%$ with $y_{t+4} - y_t \ge 15.0\%$).
   - Trend progression is sustained without immediate collapse.

---

### 4. Trend Persistence & False Spike Filtering
- **Problem**: A single isolated weekly spike (e.g. $40 \to 80 \to 39 \to 41$) does not constitute a real trend outbreak.
- **Persistence Rule**: To be classified as `Emerging Trend` (1) or `Strong Trend` (2), the future trajectory must demonstrate **sustained growth**:
  $$\text{Mean}(y_{t+1 \dots t+4}) > y_t \quad \text{AND} \quad y_{t+4} \ge y_t$$
- If a temporary spike occurs but search volume drops back by $t+4$, it is reclassified as **`Stable` (Class 0)** to prevent false alarm warnings.

---

### 5. Removal of Invalid Trailing Rows
- For the last 4 observations of each trend ($t > N - 4$), a complete 4-week lookahead window does not exist.
- $96 \text{ trends} \times 4 \text{ weeks} = 384$ trailing rows were removed.
- **Final ML Dataset Size**: 20,736 rows $\times$ 16 columns (`outputs/ml_dataset.csv`).

## 🤖 Machine Learning Methodology

### 1. Time-Aware Chronological Splitting
- **Why Time-Based Splitting is Used**: Standard random shuffling (`train_test_split`) causes temporal data leakage because future observations appear in training while past observations appear in test sets. Chronological splitting ensures the model is trained strictly on historical dates and evaluated on future holdout dates.
- **Split Breakdown (216 unique weeks)**:
  - **Train Set (70%)**: `2015-10-05` to `2018-08-20` (14,496 observations, 151 weeks)
  - **Validation Set (15%)**: `2018-08-27` to `2019-04-01` (3,072 observations, 32 weeks)
  - **Test Set (15%)**: `2019-04-08` to `2019-11-18` (3,168 observations, 33 weeks)

---

### 2. Trained Models & Class Imbalance Handling
- **Logistic Regression**: Scaled using `StandardScaler` fitted strictly on train set. Class weights set to `balanced`.
- **Random Forest Classifier**: 100 trees, `max_depth=12`, `min_samples_split=5`, `class_weight='balanced'`.
- **XGBoost Classifier**: 100 trees, `max_depth=6`, `learning_rate=0.1`, sample-weighted using balanced class weights (`compute_sample_weight`).

---

### 3. Empirical Model Evaluation Results (Holdout Test Set)

Tested on the time-aware holdout test set (3,168 observations):

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | F1-Score (Macro) | ROC-AUC (Macro OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 50.22% | 67.79% | 50.22% | 52.97% | 48.45% | 0.7513 |
| **Random Forest** | **65.09%** | **71.80%** | **65.09%** | **67.29%** | **58.47%** | **0.7993** |
| **XGBoost** | 60.07% | 72.23% | 60.07% | 63.08% | 56.06% | 0.7986 |

---

### 4. Empirical Feature Importances

Ranking of features by mean importance across models:

1. **`current_value`** (20.91%): Primary baseline popularity index.
2. **`volatility`** (15.13%): Relative coefficient of variation ($\text{std}/\text{mean}$).
3. **`moving_average_8`** (11.65%): Medium-term baseline popularity.
4. **`percentage_change`** (10.09%): Short-term percentage velocity.
5. **`absolute_change`** (9.19%): Raw weekly change magnitude.
6. **`moving_average_4`** (8.68%): Short-term baseline popularity.
7. **`previous_value`** (6.61%): Prior week search index.
8. **`trend_slope`** (5.43%): 4-week OLS linear regression slope.
9. **`rolling_std_4`** (5.39%): Standard deviation of search volume.
10. **`momentum`** (4.53%): Discrete acceleration ($y_t - 2y_{t-1} + y_{t-2}$).
11. **`persistence`** (2.38%): Consecutive growth streak counter.

## 🚨 Early Warning Engine

Converts model predictions and probabilities into actionable alerts:
- **`Stable`** $\rightarrow$ **NORMAL** (Green): No immediate action required.
- **`Emerging Trend`** $\rightarrow$ **WATCH** (Yellow): Monitor inventory & suppliers.
- **`Strong Trend`** $\rightarrow$ **ALERT** (Red): Rapid surge; increase manufacturing priority.

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Preprocessing & Feature Engineering Pipeline
```bash
python run_pipeline.py
```

### 3. Run Future Target Creation Pipeline
```bash
python create_targets.py
```

### 4. Run Machine Learning & Evaluation Pipeline
```bash
python -m src.evaluate
```

### 5. Launch Streamlit Web App
```bash
streamlit run app.py
```

The Streamlit interface will open locally at `http://localhost:8501`.

---

## 🎓 Academic Integrity & Explanation Notes
- Every calculation step is modularized in `src/`.
- No random shuffling of time-series observations to ensure strict temporal evaluation.
- All metrics and displayed predictions originate directly from trained models and empirical search volume dataset.
