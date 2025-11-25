# 🏥 Medical Insurance Cost Prediction — Machine Learning Project

This project focuses on predicting annual medical insurance charges for individuals based on demographic and health-related attributes. The dataset includes real-world healthcare patterns such as BMI, smoking status, and age — making this an excellent applied machine learning problem for regression modeling.

---

## 📌 Problem Overview

Insurance companies need to estimate medical costs for individuals in order to build fair pricing models and reduce financial risk. In this project, we use machine learning to predict the target value:


using features such as:

- Age  
- Sex  
- BMI  
- Number of children  
- Smoking status  
- Residential region  

---

## 🧠 Approach & Methodology

### ✔ 1. Exploratory Data Analysis (EDA)
- Checked data statistics and distributions
- Visualized relationships between charges & predictors
- Identified key cost-influencing attributes:
  - smoking
  - age
  - BMI

### ✔ 2. Feature Engineering
Created additional meaningful features:

- **BMI category** (underweight / normal / overweight / obese)
- **Age group** (young / adult / mid_age / senior)
- **Interaction features:**
  - age × smoker
  - age × BMI

These help the model learn non-linear relationships and improve performance.

### ✔ 3. Data Preprocessing
- Standard scaling for numerical features
- One-hot encoding for categorical features

### ✔ 4. Modeling
Models used:

- Gradient Boosting Regressor (optimized)
- XGBoost Regressor with:
  - `n_estimators=800`
  - `max_depth=5`
  - `learning_rate=0.05`
  - `subsample=0.8`

### ✔ 5. Target Transformation
To handle skewed cost distribution:

```python
y = log1p(charges)
