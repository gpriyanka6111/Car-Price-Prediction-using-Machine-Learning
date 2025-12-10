"""
Car Price Prediction Model Training Script

This script loads car sales data, preprocesses it, trains multiple regression models,
and saves the best performing model for future predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CarPricePredictor:
    """Car price prediction model trainer."""
    
    def __init__(self, data_path: str = "car sale.csv"):
        """
        Initialize the predictor.
        
        Args:
            data_path: Path to the CSV file containing car sales data
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42)
        }
        
    def load_data(self):
        """Load the car sales data from CSV."""
        logger.info(f"Loading data from {self.data_path}")
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self):
        """Clean and preprocess the data."""
        logger.info("Starting data preprocessing")
        
        # Drop unnecessary columns
        columns_to_drop = ['New_Price', 'Name', 'Location']
        # Also drop 'Unnamed: 0' if it exists
        if 'Unnamed: 0' in self.df.columns:
            columns_to_drop.append('Unnamed: 0')
        
        self.df = self.df.drop(columns_to_drop, axis=1, errors='ignore')
        logger.info(f"Dropped columns: {columns_to_drop}")
        
        # Fill missing numeric values with median
        for label, content in self.df.items():
            if pd.api.types.is_numeric_dtype(content):
                if pd.isnull(content).sum():
                    median_value = content.median()
                    self.df[label] = content.fillna(median_value)
                    logger.info(f"Filled {pd.isnull(content).sum()} missing values in '{label}' with median: {median_value}")
        
        # Convert string columns to categorical
        for label, content in self.df.items():
            if pd.api.types.is_string_dtype(content):
                self.df[label] = content.astype("category")
        
        # Convert categorical to numerical
        for label, content in self.df.items():
            if not pd.api.types.is_numeric_dtype(content):
                self.df[label] = pd.Categorical(content).codes
        
        logger.info("Data preprocessing completed")
        logger.info(f"Final shape: {self.df.shape}")
        logger.info(f"Missing values: {self.df.isna().sum().sum()}")
        
        return self.df
    
    def visualize_data(self, save_plots: bool = True):
        """
        Create visualizations of the data.
        
        Args:
            save_plots: Whether to save plots to files
        """
        logger.info("Creating data visualizations")
        
        # Correlation heatmap
        plt.figure(figsize=(15, 10))
        sns.heatmap(self.df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        if save_plots:
            plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
            logger.info("Saved correlation_heatmap.png")
        plt.close()
        
        # Pairplot (sample for performance)
        if len(self.df) > 1000:
            sample_df = self.df.sample(1000, random_state=42)
            logger.info("Creating pairplot with 1000 samples for performance")
        else:
            sample_df = self.df
        
        pairplot = sns.pairplot(sample_df)
        if save_plots:
            pairplot.savefig('feature_pairplot.png', dpi=300, bbox_inches='tight')
            logger.info("Saved feature_pairplot.png")
        plt.close()
    
    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Split data into training and testing sets.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        logger.info(f"Splitting data with test_size={test_size}")
        
        X = self.df.drop("Price", axis=1)
        y = self.df["Price"]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Training set size: {len(self.X_train)}")
        logger.info(f"Testing set size: {len(self.X_test)}")
    
    def train_models(self):
        """Train multiple models and compare their performance."""
        logger.info("Training multiple models")
        
        model_scores = {}
        cv_scores = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Test score
            test_score = model.score(self.X_test, self.y_test)
            model_scores[name] = test_score
            
            # Cross-validation score (5-fold)
            cv_score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='r2')
            cv_scores[name] = cv_score.mean()
            
            logger.info(f"{name} - Test R² Score: {test_score:.4f}, CV R² Score: {cv_score.mean():.4f}")
        
        return model_scores, cv_scores
    
    def compare_models(self, model_scores: dict, save_plot: bool = True):
        """
        Visualize model comparison.
        
        Args:
            model_scores: Dictionary of model names and their scores
            save_plot: Whether to save the plot to a file
        """
        logger.info("Creating model comparison visualization")
        
        model_compare = pd.DataFrame(model_scores, index=['R² Score'])
        
        plt.figure(figsize=(10, 6))
        model_compare.T.plot(kind='bar', legend=False, color='skyblue')
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, 1])
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
            logger.info("Saved model_comparison.png")
        plt.close()
    
    def select_best_model(self, model_scores: dict):
        """
        Select the best performing model.
        
        Args:
            model_scores: Dictionary of model names and their scores
        """
        best_model_name = max(model_scores, key=model_scores.get)
        self.model = self.models[best_model_name]
        
        # Retrain on the full training set
        self.model.fit(self.X_train, self.y_train)
        
        logger.info(f"Selected best model: {best_model_name}")
        logger.info(f"Test R² Score: {model_scores[best_model_name]:.4f}")
        
        return best_model_name
    
    def evaluate_model(self):
        """Evaluate the selected model with detailed metrics."""
        logger.info("Evaluating model performance")
        
        predictions = self.model.predict(self.X_test)
        
        mse = metrics.mean_squared_error(self.y_test, predictions)
        mae = metrics.mean_absolute_error(self.y_test, predictions)
        r2 = metrics.r2_score(self.y_test, predictions)
        rmse = np.sqrt(mse)
        
        logger.info(f"Mean Squared Error: {mse:.2f}")
        logger.info(f"Root Mean Squared Error: {rmse:.2f}")
        logger.info(f"Mean Absolute Error: {mae:.2f}")
        logger.info(f"R² Score: {r2:.4f}")
        
        # Create prediction vs actual plot
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, predictions, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 
                 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Car Prices')
        plt.tight_layout()
        plt.savefig('prediction_vs_actual.png', dpi=300, bbox_inches='tight')
        logger.info("Saved prediction_vs_actual.png")
        plt.close()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def save_model(self, filepath: str = "random_forest_regression_model.pkl"):
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path where the model should be saved
        """
        logger.info(f"Saving model to {filepath}")
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def run_full_pipeline(self):
        """Execute the complete training pipeline."""
        logger.info("Starting full training pipeline")
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Visualize data
        self.visualize_data()
        
        # Split data
        self.split_data()
        
        # Train and compare models
        model_scores, cv_scores = self.train_models()
        self.compare_models(model_scores)
        
        # Select best model
        best_model = self.select_best_model(model_scores)
        
        # Evaluate model
        metrics_dict = self.evaluate_model()
        
        # Save model
        self.save_model()
        
        logger.info("Training pipeline completed successfully")
        
        return {
            'best_model': best_model,
            'metrics': metrics_dict,
            'model_scores': model_scores,
            'cv_scores': cv_scores
        }


def main():
    """Main function to run the training pipeline."""
    predictor = CarPricePredictor()
    results = predictor.run_full_pipeline()
    
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    print(f"\nBest Model: {results['best_model']}")
    print(f"\nTest Set Metrics:")
    print(f"  R² Score: {results['metrics']['r2']:.4f}")
    print(f"  RMSE: {results['metrics']['rmse']:.2f}")
    print(f"  MAE: {results['metrics']['mae']:.2f}")
    print(f"\nAll Model Scores:")
    for model, score in results['model_scores'].items():
        cv_score = results['cv_scores'][model]
        print(f"  {model}: Test R²={score:.4f}, CV R²={cv_score:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
