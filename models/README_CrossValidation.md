# Cross-Validation Implementation for MLB Playoff Prediction Models

## Overview

This directory contains enhanced versions of the original models with comprehensive cross-validation using GridSearchCV and RandomizedSearchCV. The implementation provides robust hyperparameter tuning, performance evaluation, and model comparison capabilities.

## Files Added

### Core Cross-Validation Models
- `logistic_model_cv.py` - Logistic Regression with GridSearchCV
- `xgb_model_cv.py` - XGBoost with RandomizedSearchCV  
- `model_utils_cv.py` - Enhanced utilities for cross-validated models
- `run_cv_training.py` - Pipeline script to train both models
- `demo_cv_models.py` - Demonstration of model usage

### Generated Assets
- `assets/logistic_playoffs_cv.pkl` - Trained logistic model with CV metadata
- `assets/logistic_playoffs_cv_summary.json` - Human-readable summary
- `assets/xgb_playoffs_cv.pkl` - Trained XGBoost model with CV metadata
- `assets/xgb_playoffs_cv_summary.json` - Human-readable summary
- `assets/cv_model_comparison_report.json` - Comprehensive comparison report

## Key Improvements Over Original Models

### 1. Systematic Hyperparameter Tuning

**Logistic Regression (GridSearchCV)**:
- **C**: Regularization strength [0.01, 0.1, 1.0, 10.0, 100.0]
- **penalty**: L1, L2, and ElasticNet regularization
- **solver**: liblinear, saga (supports all penalty types)
- **class_weight**: None vs. balanced for handling class imbalance
- **max_iter**: [1000, 2000] for convergence
- **l1_ratio**: [0.1, 0.5, 0.9] for ElasticNet penalty

**XGBoost (RandomizedSearchCV)**:
- **learning_rate**: 0.01 to 0.31 (continuous uniform)
- **max_depth**: 3 to 9 (discrete uniform)
- **min_child_weight**: 1 to 5
- **reg_alpha**: 0 to 1 (L1 regularization)
- **reg_lambda**: 1 to 5 (L2 regularization)
- **subsample**: 0.6 to 1.0 (row sampling)
- **colsample_bytree**: 0.6 to 1.0 (feature sampling per tree)
- **colsample_bylevel**: 0.6 to 1.0 (feature sampling per level)
- **n_estimators**: 100 to 999
- **gamma**: 0 to 0.5 (minimum split loss)

### 2. Robust Cross-Validation Strategy

- **StratifiedKFold**: 5-fold CV maintaining class distribution
- **Multiple Metrics**: ROC AUC, accuracy, precision, recall, F1-score
- **Stability Analysis**: Coefficient of variation and performance consistency
- **Early Stopping**: For XGBoost to prevent overfitting

### 3. Comprehensive Performance Analysis

**Model Comparison Features**:
- Cross-validation score comparison
- Training time analysis
- Feature importance comparison
- Prediction agreement analysis
- Ensemble prediction capabilities
- ROC curve analysis
- Model stability metrics

## Performance Results

### Cross-Validation Scores (5-fold)
- **Logistic Regression**: 0.9775 ± 0.0119 ROC AUC
- **XGBoost**: 0.9725 ± 0.0248 ROC AUC

### Validation Performance (2024 data)
- **Logistic Regression**: 0.9444 ROC AUC, 86.7% accuracy
- **XGBoost**: 0.9676 ROC AUC, 90.0% accuracy

### Training Efficiency
- **Logistic Regression**: 13.09 seconds, 120 parameter combinations
- **XGBoost**: 6.86 seconds, 100 parameter combinations

### Key Findings

1. **Logistic Regression** achieved slightly better cross-validation performance
2. **XGBoost** showed better validation performance and faster training
3. Models agree on **96.7%** of predictions
4. Both models show excellent stability (low coefficient of variation)

## Best Hyperparameters Found

### Logistic Regression
```json
{
  "C": 10.0,
  "class_weight": null,
  "max_iter": 1000,
  "penalty": "l2",
  "solver": "saga"
}
```

### XGBoost
```json
{
  "learning_rate": 0.0620,
  "max_depth": 8,
  "min_child_weight": 5,
  "n_estimators": 122,
  "reg_alpha": 0.0481,
  "reg_lambda": 4.7966,
  "subsample": 0.9547,
  "colsample_bytree": 0.7966,
  "colsample_bylevel": 0.6391,
  "gamma": 0.2367
}
```

## Feature Importance Analysis

### Most Important Features (Both Models)
1. **P_L** (Pitching Losses) - Most predictive feature
2. **P_W** (Pitching Wins) - Strong positive indicator
3. **P_ERA** (Earned Run Average) - Key pitching metric
4. **P_R** (Runs Allowed) - Defensive performance
5. **P_AVG** (Opponent Batting Average) - Pitching effectiveness

### Model Disagreements
The models show the most disagreement on:
- **P_H** (Hits Allowed): XGB weights lower, Logistic higher
- **P_SV** (Saves): XGB weights much higher
- **H_HR** (Home Runs): Logistic weights much higher

## Usage Examples

### Running Complete Pipeline
```bash
# Train both models with cross-validation
python models/run_cv_training.py

# Demonstrate model capabilities
python models/demo_cv_models.py
```

### Loading Models Programmatically
```python
from models.model_utils_cv import load_both_cv_models, get_cv_model_predictions

# Load both models
xgb_data, logistic_data = load_both_cv_models()

# Make predictions
predictions = get_cv_model_predictions(X, xgb_data, logistic_data)

# Create ensemble
ensemble = get_ensemble_cv_prediction(predictions, method="average")
```

### Model Comparison
```python
from models.model_utils_cv import compare_cv_training_results

comparison = compare_cv_training_results()
print(f"Better model: {comparison['better_model']}")
print(f"Score difference: {comparison['score_difference']:.4f}")
```

## Cross-Validation Methodology

### Grid Search (Logistic Regression)
- **Exhaustive search** over parameter combinations
- **120 total combinations** tested
- **Parallel processing** for efficiency
- **Return train scores** for overfitting analysis

### Randomized Search (XGBoost)  
- **Random sampling** from parameter distributions
- **100 random combinations** tested
- **Continuous and discrete distributions** appropriately handled
- **Faster than grid search** for high-dimensional spaces

### Validation Strategy
- **Time-based split**: Train on ≤2023, validate on 2024+
- **Stratified CV**: Maintains class balance in each fold
- **Multiple metrics**: Comprehensive performance evaluation
- **Stability analysis**: Coefficient of variation tracking

## Model Stability Analysis

### XGBoost Stability
- Mean CV Score: 0.9670 ± 0.0030
- Coefficient of Variation: 0.0031 (very stable)
- Score Range: 0.0144
- High Performers: 10/100 combinations

### Logistic Regression Stability  
- Mean CV Score: 0.9744 ± 0.0055
- Coefficient of Variation: 0.0057 (stable)
- Score Range: 0.0198
- High Performers: 8/120 combinations

## Ensemble Capabilities

The implementation supports multiple ensemble strategies:

### Average Ensemble
```python
ensemble = get_ensemble_cv_prediction(predictions, method="average")
```

### Weighted Ensemble  
```python
ensemble = get_ensemble_cv_prediction(
    predictions, 
    method="weighted", 
    weights=[0.6, 0.4]  # 60% XGBoost, 40% Logistic
)
```

## Files Structure

```
models/
├── logistic_model_cv.py          # Enhanced logistic regression
├── xgb_model_cv.py               # Enhanced XGBoost  
├── model_utils_cv.py             # Cross-validation utilities
├── run_cv_training.py            # Training pipeline
├── demo_cv_models.py             # Usage demonstration
├── logistic_model.py             # Original logistic model
├── xgb_model.py                  # Original XGBoost model
└── model_utils.py                # Original utilities

assets/
├── logistic_playoffs_cv.pkl      # CV logistic model
├── logistic_playoffs_cv_summary.json
├── xgb_playoffs_cv.pkl           # CV XGBoost model  
├── xgb_playoffs_cv_summary.json
├── cv_model_comparison_report.json
├── logistic_playoffs.pkl         # Original models
└── xgb_playoffs.pkl
```

## Next Steps and Recommendations

### Model Deployment
1. Use **ensemble predictions** for best performance
2. Monitor **prediction agreement** for confidence assessment
3. Implement **prediction intervals** using CV results
4. Set up **automated retraining** pipeline

### Further Improvements
1. **Nested CV** for unbiased performance estimation
2. **Bayesian optimization** for XGBoost hyperparameters  
3. **Feature selection** using CV results
4. **Time series validation** for temporal dependencies
5. **Cost-sensitive learning** for business metrics

### Production Considerations
- Model versioning and metadata tracking
- A/B testing framework for model comparison
- Performance monitoring and drift detection
- Automated hyperparameter retuning schedule

## Summary

The cross-validation implementation provides:

✅ **Robust hyperparameter tuning** with systematic search strategies  
✅ **Comprehensive performance evaluation** across multiple metrics  
✅ **Model stability analysis** for production confidence  
✅ **Ensemble capabilities** for improved predictions  
✅ **Detailed comparison framework** for model selection  
✅ **Production-ready utilities** for easy deployment  

Both models achieve excellent performance with ROC AUC scores above 0.97, demonstrating the effectiveness of the cross-validation approach for this MLB playoff prediction task.