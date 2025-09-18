from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import warnings
from contextlib import redirect_stderr
import io

class EnsemblePredictor:
    """Enhanced ensemble predictor combining multiple algorithms"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=500,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=-1,
                random_state=42
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=300,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'ridge': Ridge(
                alpha=1.0,
                random_state=42
            )
        }
        self.weights = {'rf': 0.5, 'gb': 0.3, 'ridge': 0.2}
        self.feature_importance_ = None
        
    def fit(self, X_train, y_train):
        """Train all models in the ensemble"""
        # Fill any NaN values before training
        X_train_filled = X_train.fillna(0)
        
        # Suppress sklearn warnings during training
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
            warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy')
            
            for name, model in self.models.items():
                model.fit(X_train_filled, y_train)
        
        # Calculate feature importance from Random Forest
        self.feature_importance_ = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.models['rf'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self
    
    def predict(self, X):
        """Make predictions using weighted ensemble"""
        # Fill any remaining NaN values before prediction
        X_filled = X.fillna(0)
        
        predictions = np.zeros(len(X_filled))
        
        # Suppress sklearn warnings during prediction
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
            
            for name, model in self.models.items():
                pred = model.predict(X_filled)
                predictions += self.weights[name] * pred
                
        return predictions
    
    def get_feature_importance(self):
        """Get feature importance from the random forest model"""
        return self.feature_importance_

def train_model(X_train, y_train):
    """Train an enhanced ensemble model"""
    
    model = EnsemblePredictor()
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test):
    """Enhanced model evaluation with additional metrics"""

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    signs_correct = np.sum(np.sign(y_test) == np.sign(y_pred))
    signs_percent = (signs_correct / len(y_test)) * 100
    
    # Additional metrics for better evaluation
    # Handle division by zero in MAPE calculation
    non_zero_mask = np.abs(y_test) > 1e-8
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs((y_test[non_zero_mask] - y_pred[non_zero_mask]) / y_test[non_zero_mask])) * 100
    else:
        mape = float('inf')  # Set to infinity if all actual values are zero
    
    # Prediction accuracy in different ranges
    small_changes = np.abs(y_test) < 50000  # Small changes < 50k
    medium_changes = (np.abs(y_test) >= 50000) & (np.abs(y_test) < 200000)  # Medium changes 50k-200k
    large_changes = np.abs(y_test) >= 200000  # Large changes > 200k
    
    small_acc = np.mean(np.sign(y_test[small_changes]) == np.sign(y_pred[small_changes])) * 100 if np.sum(small_changes) > 0 else 0
    medium_acc = np.mean(np.sign(y_test[medium_changes]) == np.sign(y_pred[medium_changes])) * 100 if np.sum(medium_changes) > 0 else 0
    large_acc = np.mean(np.sign(y_test[large_changes]) == np.sign(y_pred[large_changes])) * 100 if np.sum(large_changes) > 0 else 0

    return signs_percent, rmse, mae, r2, mape, small_acc, medium_acc, large_acc


def cross_validate_model(X, y, cv=5):
    """Perform cross-validation on the model"""
    
    model = EnsemblePredictor()
    
    # Cross-validation scores with warning suppression
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        cv_scores = cross_val_score(model.models['rf'], X, y, cv=cv, scoring='neg_mean_squared_error')
    
    cv_rmse = np.sqrt(-cv_scores)
    
    return cv_rmse.mean(), cv_rmse.std()


def get_prediction_confidence(model, X, n_estimators_subset=50):
    """Calculate prediction confidence using Random Forest variance"""
    
    # Get predictions from individual trees for confidence estimation
    rf_model = model.models['rf']
    
    # Suppress warnings during confidence calculation
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        tree_predictions = np.array([tree.predict(X) for tree in rf_model.estimators_[:n_estimators_subset]])
    
    # Calculate prediction statistics
    pred_mean = np.mean(tree_predictions, axis=0)
    pred_std = np.std(tree_predictions, axis=0)
    
    # Confidence intervals (95%)
    confidence_lower = pred_mean - 1.96 * pred_std
    confidence_upper = pred_mean + 1.96 * pred_std
    
    return pred_std, confidence_lower, confidence_upper