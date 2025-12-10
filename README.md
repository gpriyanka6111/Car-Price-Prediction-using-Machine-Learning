# 🚗 Car Price Prediction using Machine Learning

This project predicts the **selling price of cars** using multiple regression models in Python.  
It includes **data cleaning, feature engineering, visualization, model comparison, and evaluation**, with the final model saved as a `.pkl` file for reuse.

---

## 📊 Features
- **Data Preprocessing**
  - Handles missing values (numeric → median, categorical → category codes)
  - Drops irrelevant columns (e.g., `Name`, `Location`, `New_Price`)
  - Converts categorical features to numerical values
- **Exploratory Data Analysis (EDA)**
  - Correlation heatmaps
  - Pairplots for feature relationships
  - Automated visualization generation
- **Model Training & Evaluation**
  - Compares **Linear Regression**, **Decision Tree**, and **Random Forest**
  - Implements **cross-validation** for robust model assessment
  - Evaluates with MSE, RMSE, MAE, and R² score
  - Generates prediction vs actual plots
- **Model Persistence**
  - Saves trained model with **pickle** for future predictions
  - Includes prediction script for easy reuse
- **Logging & Error Handling**
  - Comprehensive logging for debugging
  - Robust error handling throughout

---

## 🧰 Tech Stack
- **Python 3.x**
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, Pickle

---

## 🚀 Quick Start

**New here?** Check out the [Quick Start Guide](QUICKSTART.md) for a 5-minute introduction!

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/gpriyanka6111/Car-Price-Prediction-using-Machine-Learning.git
cd Car-Price-Prediction-using-Machine-Learning
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train_model.py
```

---

## 📖 Usage

### Training the Model

Run the training script to preprocess data, train models, and save the best model:

```bash
python train_model.py
```

This will:
- Load and preprocess the car sales data
- Generate visualization plots (saved as PNG files)
- Train and compare multiple models
- Save the best model as `random_forest_regression_model.pkl`
- Display comprehensive performance metrics

### Making Predictions

Use the prediction script with a trained model:

```bash
# Make predictions from a CSV file
python predict.py --input your_data.csv --output predictions.csv

# Use a specific model file
python predict.py --model mymodel.pkl --input data.csv
```

### Using the Jupyter Notebook

For interactive exploration:
```bash
jupyter notebook "Car Price Prediction.ipynb"
```

---

## 📁 Project Structure

```
.
├── car sale.csv                          # Dataset
├── Car Price Prediction.ipynb            # Jupyter notebook (original)
├── Car Price Prediction.py               # Original Python script
├── train_model.py                        # Modern training script
├── predict.py                            # Prediction script
├── requirements.txt                      # Python dependencies
├── .gitignore                           # Git ignore file
└── README.md                            # Project documentation
```

---

## 📊 Model Performance

The Random Forest Regressor typically achieves:
- **R² Score**: ~0.84 (84% of variance explained)
- **RMSE**: ~4.38
- **MAE**: ~1.77

---

## 🔄 Recent Updates

- ✅ Added modular, object-oriented code structure
- ✅ Implemented comprehensive logging
- ✅ Added cross-validation for model evaluation
- ✅ Created command-line interface for predictions
- ✅ Enhanced error handling and documentation
- ✅ Added requirements.txt for dependency management
- ✅ Improved visualization generation
- ✅ Added .gitignore for cleaner repository

---
