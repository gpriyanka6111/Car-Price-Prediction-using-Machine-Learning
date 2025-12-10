"""
Car Price Prediction Script

This script loads a trained model and makes predictions on new data.
"""

import pickle
import argparse
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CarPriceEstimator:
    """Car price estimator using a pre-trained model."""
    
    def __init__(self, model_path: str = "random_forest_regression_model.pkl"):
        """
        Initialize the estimator.
        
        Args:
            model_path: Path to the pickled model file
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model from file."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict(self, features: pd.DataFrame) -> float:
        """
        Make a price prediction.
        
        Args:
            features: DataFrame with car features
            
        Returns:
            Predicted price
        """
        try:
            prediction = self.model.predict(features)
            return prediction[0]
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_from_csv(self, csv_path: str, output_path: str = None):
        """
        Make predictions for cars in a CSV file.
        
        Args:
            csv_path: Path to input CSV file
            output_path: Path to save predictions (optional)
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Loading data from {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            predictions = self.model.predict(df)
            df['Predicted_Price'] = predictions
            
            logger.info(f"Made {len(predictions)} predictions")
            
            if output_path:
                df.to_csv(output_path, index=False)
                logger.info(f"Saved predictions to {output_path}")
            
            return df
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Predict car prices using a trained model'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest_regression_model.pkl',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input CSV file with car features'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save predictions (optional)'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first by running: python train_model.py")
        return
    
    # Initialize estimator
    estimator = CarPriceEstimator(args.model)
    
    if args.input:
        # Predict from CSV
        df = estimator.predict_from_csv(args.input, args.output)
        print(f"\nPredictions made for {len(df)} cars")
        print(f"\nSample predictions:")
        print(df.head())
    else:
        print("\nCar Price Estimator")
        print("==================")
        print(f"Model loaded: {args.model}")
        print("\nUsage examples:")
        print("  python predict.py --input data.csv --output predictions.csv")
        print("  python predict.py --model mymodel.pkl --input data.csv")


if __name__ == "__main__":
    main()
