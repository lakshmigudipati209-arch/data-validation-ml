# Ad Click-Through Prediction — Data Validation & Logistic Regression

Predicting whether a user clicks on an online ad using demographic and behavioral features, with a focus on proper data validation and cross-validation techniques.

## 📌 Overview

This project uses the **Advertising dataset** (1000 records) to:
- Validate data quality (check for missing/null values)
- Build a baseline **Logistic Regression** classifier
- Evaluate model performance using train-test split and cross-validation

## 📊 Dataset

Features used:
| Feature | Description |
|---|---|
| Daily Time Spent on Site | Minutes spent on site |
| Age | User age |
| Area Income | Average income of user's area |
| Daily Internet Usage | Minutes spent on internet daily |
| Male | Gender (0/1) |
| Ad Topic Line, City, Country, Timestamp | Metadata (not used as model features) |

**Target:** `Clicked on Ad` (0 = No, 1 = Yes)

## 🔍 Data Validation

- Checked the full dataset with `df.isnull()` to confirm no missing values across all 10 columns before modeling.

## 🧠 Modeling Approach

1. **Train-Test Split**
   - 70% train / 30% test, stratified on target
   - `train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)`

2. **Baseline Logistic Regression**
   - Training Accuracy: **0.521**
   - Test Accuracy: **0.493**

3. **Cross-Validation (K-Fold + ShuffleSplit)**
   - `LogisticRegression(solver='liblinear')`
   - ShuffleSplit with 200 splits, 30% test size
   - Cross-validated Train Accuracy: **0.54**
   - Cross-validated Test Accuracy: **0.51**

## ✅ Key Takeaway

A single train-test split can be misleading — cross-validation gives a more reliable estimate of model performance. The baseline model shows there's room for improvement through feature engineering, encoding categorical variables (Ad Topic, City, Country), and trying non-linear models.

## 🛠️ Tech Stack

- Python, pandas, numpy
- scikit-learn (`LogisticRegression`, `train_test_split`, `KFold`, `ShuffleSplit`, `cross_validate`)


## 🚀 Next Steps

- Encode categorical features
- Try Random Forest / XGBoost for comparison
- Feature scaling and hyperparameter tuning
