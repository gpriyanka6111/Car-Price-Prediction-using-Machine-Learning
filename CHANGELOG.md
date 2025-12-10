# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2025-12-10

### Added
- **New modular training script** (`train_model.py`) with object-oriented design
- **Prediction script** (`predict.py`) with command-line interface
- **Example usage script** (`example_usage.py`) for demonstrations
- **Requirements file** (`requirements.txt`) for easy dependency management
- **Comprehensive logging** throughout all scripts
- **Cross-validation** for more robust model evaluation
- **Additional metrics**: RMSE added alongside MSE, MAE, and R²
- **Visualization outputs**:
  - Correlation heatmap
  - Feature pairplot
  - Model comparison bar chart
  - Prediction vs actual scatter plot
- **Error handling** for file operations and model loading
- **Type hints** in function signatures
- **Comprehensive docstrings** for all classes and functions
- **Git ignore file** (`.gitignore`) to exclude artifacts
- **MIT License** for open-source distribution
- **Enhanced README** with installation and usage instructions

### Improved
- **Code structure**: Moved from script-based to class-based architecture
- **Model evaluation**: Added cross-validation scores for better assessment
- **Documentation**: Complete rewrite with examples and usage instructions
- **Visualization**: Automated generation with proper file naming
- **Performance**: Added sampling for pairplot to handle large datasets

### Changed
- Renamed output model file to include algorithm name for clarity
- Restructured code into reusable functions and classes
- Updated plotting to save figures instead of displaying interactively

### Technical Improvements
- Followed PEP 8 style guidelines
- Added logging for debugging and monitoring
- Implemented proper exception handling
- Created modular, testable code structure
- Added feature importance analysis
- Improved random state handling for reproducibility

## [1.0.0] - Original Release

### Initial Features
- Basic car price prediction using Jupyter notebook
- Data preprocessing and cleaning
- Multiple model comparison (Linear Regression, Decision Tree, Random Forest)
- Model persistence with pickle
- Basic visualization (heatmap and pairplot)
- Simple Python script version
