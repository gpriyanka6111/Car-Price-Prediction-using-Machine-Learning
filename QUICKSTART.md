# Quick Start Guide

This guide will help you get started with the Car Price Prediction project in just a few minutes.

## Prerequisites

- Python 3.7 or higher
- pip package manager

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gpriyanka6111/Car-Price-Prediction-using-Machine-Learning.git
   cd Car-Price-Prediction-using-Machine-Learning
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

### 1. Train the Model

Run the training script to build and save the model:

```bash
python train_model.py
```

**Expected output:**
- Console logs showing training progress
- Generated visualization files (*.png)
- Saved model file (`random_forest_regression_model.pkl`)
- Performance metrics (R² score, RMSE, MAE)

**Time required:** ~30-60 seconds

### 2. View Example Usage

See how to use the trained model:

```bash
python example_usage.py
```

This demonstrates:
- Loading a trained model
- Making predictions
- Feature importance analysis

### 3. Interactive Exploration (Optional)

For interactive data exploration and experimentation:

```bash
jupyter notebook "Car Price Prediction.ipynb"
```

## What You Get

After running `train_model.py`, you'll have:

1. **Trained Model** (`*.pkl`) - Ready for predictions
2. **Visualizations:**
   - `correlation_heatmap.png` - Feature correlations
   - `feature_pairplot.png` - Pairwise relationships
   - `model_comparison.png` - Model performance comparison
   - `prediction_vs_actual.png` - Prediction accuracy visualization

## Next Steps

- Review the [README.md](README.md) for detailed documentation
- Check [CHANGELOG.md](CHANGELOG.md) for recent updates
- Explore the code in `train_model.py` to understand the pipeline
- Modify hyperparameters in the code for experimentation

## Troubleshooting

**Issue:** Import errors
- **Solution:** Make sure all dependencies are installed: `pip install -r requirements.txt`

**Issue:** File not found error
- **Solution:** Ensure `car sale.csv` is in the project directory

**Issue:** Memory issues with visualizations
- **Solution:** The script automatically samples large datasets for pairplots

## Project Structure

```
.
├── train_model.py          # Main training script
├── predict.py              # Prediction script
├── example_usage.py        # Usage examples
├── car sale.csv            # Dataset
├── requirements.txt        # Dependencies
└── README.md              # Full documentation
```

## Key Features

✅ **Automated Training Pipeline** - One command to train and evaluate models  
✅ **Cross-Validation** - Robust model assessment  
✅ **Multiple Models** - Compares Linear Regression, Decision Tree, Random Forest  
✅ **Comprehensive Logging** - Track training progress  
✅ **Visualization** - Automatic generation of analysis plots  
✅ **Model Persistence** - Save and reuse trained models  

## Performance

Typical results with the provided dataset:
- **R² Score:** ~0.84 (84% variance explained)
- **RMSE:** ~4.38
- **MAE:** ~1.77

---

For more information, see the [full README](README.md).
