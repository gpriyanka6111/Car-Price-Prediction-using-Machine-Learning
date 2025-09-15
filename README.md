# 🚗 Car Price Prediction using Machine Learning

This project predicts the **selling price of cars** using multiple regression models in Python.  
It includes **data cleaning, feature engineering, visualization, model comparison, and evaluation**, with the final model saved as a `.pkl` file for reuse.

---

## 📊 Features
- **Data Preprocessing**
  - Handled missing values (numeric → median, categorical → category codes).
  - Dropped irrelevant columns (e.g., `Name`, `Location`, `New_Price`).
  - Converted categorical features to numerical values.
- **Exploratory Data Analysis (EDA)**
  - Correlation heatmaps
  - Pairplots for feature relationships
- **Model Training & Evaluation**
  - Compared **Linear Regression**, **Decision Tree**, and **Random Forest**.
  - Chose **Random Forest Regressor** for best accuracy.
  - Evaluated with MSE, MAE, and R² score.
- **Model Persistence**
  - Saved trained Random Forest model with **pickle** for future predictions.

---

## 🧰 Tech Stack
- **Python 3.x**
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, Pickle

---
