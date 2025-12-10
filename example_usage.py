"""
Example Usage Script

This script demonstrates how to use the trained model for predictions.
"""

import pickle
import pandas as pd
import numpy as np


def load_model(model_path='random_forest_regression_model.pkl'):
    """Load the trained model."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def example_prediction():
    """Example of making a single prediction."""
    print("="*60)
    print("Example: Car Price Prediction")
    print("="*60)
    
    # Load the model
    try:
        model = load_model()
        print("\n✓ Model loaded successfully!")
    except FileNotFoundError:
        print("\n✗ Model file not found!")
        print("  Please run 'python train_model.py' first to train the model.")
        return
    
    # Example car features (after preprocessing)
    # These are sample values - in practice, you'd need to preprocess your data
    # the same way the training data was preprocessed
    print("\nNote: This example uses preprocessed feature values.")
    print("In practice, you need to preprocess raw car data the same way")
    print("the training data was preprocessed.\n")
    
    # Create example feature data
    example_features = pd.DataFrame({
        'Year': [2015],
        'Kilometers_Driven': [50000],
        'Fuel_Type': [1],  # Encoded categorical
        'Transmission': [0],  # Encoded categorical
        'Owner_Type': [0],  # Encoded categorical
        'Mileage': [18.5],
        'Engine': [1500],
        'Power': [100],
        'Seats': [5]
    })
    
    print("Input Features:")
    print(example_features.to_string(index=False))
    
    # Make prediction
    predicted_price = model.predict(example_features)
    
    print(f"\n{'='*60}")
    print(f"Predicted Price: ₹{predicted_price[0]:.2f} Lakhs")
    print(f"{'='*60}\n")


def batch_prediction_example():
    """Example of making batch predictions from a CSV."""
    print("\n" + "="*60)
    print("Batch Prediction Example")
    print("="*60)
    
    print("\nTo make predictions on multiple cars:")
    print("1. Prepare a CSV file with the same preprocessed features")
    print("2. Run: python predict.py --input your_data.csv --output predictions.csv")
    print("\nThe features should match those used in training:")
    print("  - Year, Kilometers_Driven, Fuel_Type (encoded)")
    print("  - Transmission (encoded), Owner_Type (encoded)")
    print("  - Mileage, Engine, Power, Seats")
    print()


def show_model_info():
    """Display information about the trained model."""
    try:
        model = load_model()
        
        print("\n" + "="*60)
        print("Model Information")
        print("="*60)
        print(f"\nModel Type: {type(model).__name__}")
        print(f"Number of Trees: {model.n_estimators}")
        print(f"Number of Features: {model.n_features_in_}")
        
        if hasattr(model, 'feature_importances_'):
            print("\nTop 5 Most Important Features:")
            # Try to get feature names from the model, fallback to hardcoded list
            if hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
            else:
                # Note: These are the expected feature names after preprocessing
                feature_names = [
                    'Year', 'Kilometers_Driven', 'Fuel_Type',
                    'Transmission', 'Owner_Type', 'Mileage',
                    'Engine', 'Power', 'Seats'
                ]
            
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:5]
            
            for i, idx in enumerate(indices, 1):
                print(f"  {i}. {feature_names[idx]}: {importances[idx]:.4f}")
        print()
        
    except FileNotFoundError:
        print("\nModel file not found. Please train the model first.")


def main():
    """Main function."""
    print("\n" + "="*60)
    print("Car Price Prediction - Example Usage")
    print("="*60)
    
    # Show model information
    show_model_info()
    
    # Example single prediction
    example_prediction()
    
    # Batch prediction example
    batch_prediction_example()
    
    print("="*60)
    print("For more information, see the README.md file")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
