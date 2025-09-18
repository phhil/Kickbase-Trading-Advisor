from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import warnings
from contextlib import redirect_stderr
import io
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from numba import jit

class AdvancedEnsemblePredictor:
    """Enhanced ensemble predictor with advanced gradient boosting models and specialized handling for different value ranges"""
    
    def __init__(self, fast_mode=False):
        self.fast_mode = fast_mode
        
        if fast_mode:
            # Faster training with reduced complexity for development/testing
            self.models = {
                'xgb': xgb.XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1
                ),
                'lgb': lgb.LGBMRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                'rf': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=42
                )
            }
            # Simplified weights for fast mode
            self.weights = {'xgb': 0.4, 'lgb': 0.4, 'rf': 0.2}
        else:
            # Advanced models with optimized parameters for financial predictions
            self.models = {
                'xgb': xgb.XGBRegressor(
                    n_estimators=1000,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    early_stopping_rounds=50,
                    random_state=42,
                    n_jobs=-1
                ),
                'lgb': lgb.LGBMRegressor(
                    n_estimators=1000,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    early_stopping_rounds=50,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ),
                'cat': cb.CatBoostRegressor(
                    iterations=1000,
                    depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    reg_lambda=1.0,
                    early_stopping_rounds=50,
                    random_state=42,
                    verbose=False,
                    thread_count=-1
                ),
                'rf': RandomForestRegressor(
                    n_estimators=500,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=42
                ),
                'gb': GradientBoostingRegressor(
                    n_estimators=500,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42
                ),
                'elastic': ElasticNet(
                    alpha=0.1,
                    l1_ratio=0.5,
                    random_state=42
                )
            }
            
            # Dynamic weights - give more weight to advanced models
            self.weights = {
                'xgb': 0.25,
                'lgb': 0.25, 
                'cat': 0.20,
                'rf': 0.15,
                'gb': 0.10,
                'elastic': 0.05
            }
        
        # Specialized models for different value ranges
        self.range_models = {
            'small': None,   # For changes < 50k
            'medium': None,  # For changes 50k-200k  
            'large': None    # For changes > 200k
        }
        
        self.feature_importance_ = None
        self.is_fitted = False
        
    def _create_range_specific_model(self, model_type='xgb'):
        """Create models optimized for specific value ranges"""
        if model_type == 'xgb':
            return xgb.XGBRegressor(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.2,
                reg_lambda=1.5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'lgb':
            return lgb.LGBMRegressor(
                n_estimators=800,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.2,
                reg_lambda=1.5,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
    def fit(self, X_train, y_train):
        """Train all models in the ensemble with specialized handling"""
        # Fill any NaN values before training
        X_train_filled = X_train.fillna(0)
        
        print("Training advanced ensemble models...")
        
        # Suppress warnings during training
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            # Train main ensemble models
            for name, model in self.models.items():
                print(f"Training {name.upper()}...")
                if name in ['xgb', 'lgb']:
                    # Use validation for early stopping
                    split_idx = int(0.8 * len(X_train_filled))
                    X_val = X_train_filled[split_idx:]
                    y_val = y_train[split_idx:]
                    X_train_sub = X_train_filled[:split_idx]
                    y_train_sub = y_train[:split_idx]
                    
                    if name == 'xgb':
                        model.fit(X_train_sub, y_train_sub, 
                                eval_set=[(X_val, y_val)], verbose=False)
                    elif name == 'lgb':
                        model.fit(X_train_sub, y_train_sub, 
                                eval_set=[(X_val, y_val)])
                elif name == 'cat':
                    # CatBoost handles validation internally
                    split_idx = int(0.8 * len(X_train_filled))
                    X_val = X_train_filled[split_idx:]
                    y_val = y_train[split_idx:]
                    X_train_sub = X_train_filled[:split_idx]
                    y_train_sub = y_train[:split_idx]
                    
                    model.fit(X_train_sub, y_train_sub, 
                            eval_set=(X_val, y_val))
                else:
                    model.fit(X_train_filled, y_train)
        
        # Train range-specific models for small changes (most challenging)
        print("Training specialized models for small value changes...")
        
        # Create masks for different ranges
        small_mask = np.abs(y_train) < 50000
        medium_mask = (np.abs(y_train) >= 50000) & (np.abs(y_train) < 200000)
        large_mask = np.abs(y_train) >= 200000
        
        # Train specialized model for small changes if we have enough data
        if np.sum(small_mask) > 100:
            self.range_models['small'] = self._create_range_specific_model('lgb')  # LightGBM often better for small patterns
            self.range_models['small'].fit(X_train_filled[small_mask], y_train[small_mask])
            
        if np.sum(medium_mask) > 100:
            self.range_models['medium'] = self._create_range_specific_model('xgb')
            self.range_models['medium'].fit(X_train_filled[medium_mask], y_train[medium_mask])
            
        if np.sum(large_mask) > 100:
            self.range_models['large'] = self._create_range_specific_model('xgb')
            self.range_models['large'].fit(X_train_filled[large_mask], y_train[large_mask])
        
        # Calculate feature importance from XGBoost (usually most reliable)
        if 'xgb' in self.models:
            feature_names = X_train.columns if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
            self.feature_importance_ = pd.DataFrame({
                'feature': feature_names,
                'importance': self.models['xgb'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        self.is_fitted = True
        print("Advanced ensemble training completed!")
        return self
    
    def predict(self, X):
        """Make predictions using weighted ensemble and range-specific models"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Fill any remaining NaN values before prediction
        X_filled = X.fillna(0)
        
        # Get base ensemble predictions
        predictions = np.zeros(len(X_filled))
        
        # Suppress warnings during prediction
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            for name, model in self.models.items():
                if hasattr(model, 'predict'):
                    pred = model.predict(X_filled)
                    predictions += self.weights[name] * pred
        
        # Apply range-specific adjustments
        # This is where we can improve small value change predictions
        if self.range_models['small'] is not None:
            # For predicted small changes, blend with specialized model
            small_pred_mask = np.abs(predictions) < 75000  # Slightly larger threshold for prediction
            if np.any(small_pred_mask):
                small_specialized = self.range_models['small'].predict(X_filled[small_pred_mask])
                # Blend: 70% specialized model, 30% ensemble for small changes
                predictions[small_pred_mask] = 0.7 * small_specialized + 0.3 * predictions[small_pred_mask]
        
        return predictions
    
    def get_feature_importance(self):
        """Get feature importance from the XGBoost model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        return self.feature_importance_

def train_model(X_train, y_train, fast_mode=False):
    """Train an advanced ensemble model with better performance"""
    
    model = AdvancedEnsemblePredictor(fast_mode=fast_mode)
    model.fit(X_train, y_train)
    
    return model


@jit(nopython=True)
def fast_sign_accuracy(y_true, y_pred, threshold_low, threshold_high):
    """Fast computation of sign accuracy for different value ranges using numba"""
    total_correct = 0
    total_count = 0
    
    for i in range(len(y_true)):
        if threshold_low <= abs(y_true[i]) < threshold_high:
            if (y_true[i] >= 0 and y_pred[i] >= 0) or (y_true[i] < 0 and y_pred[i] < 0):
                total_correct += 1
            total_count += 1
    
    return total_correct / max(total_count, 1) * 100

@jit(nopython=True)
def fast_mape_calculation(y_true, y_pred, min_threshold=1e-8):
    """Fast MAPE calculation with numba optimization"""
    total_ape = 0.0
    count = 0
    
    for i in range(len(y_true)):
        if abs(y_true[i]) > min_threshold:
            ape = abs((y_true[i] - y_pred[i]) / y_true[i])
            total_ape += ape
            count += 1
    
    return (total_ape / max(count, 1)) * 100 if count > 0 else float('inf')


def evaluate_model(model, X_test, y_test):
    """Enhanced model evaluation with additional metrics and performance optimizations"""

    y_pred = model.predict(X_test)

    # Basic metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    signs_correct = np.sum(np.sign(y_test) == np.sign(y_pred))
    signs_percent = (signs_correct / len(y_test)) * 100
    
    # Fast MAPE calculation using numba
    mape = fast_mape_calculation(y_test.values if hasattr(y_test, 'values') else y_test, y_pred)
    
    # Fast prediction accuracy in different ranges using numba
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    small_acc = fast_sign_accuracy(y_test_np, y_pred, 0, 50000)
    medium_acc = fast_sign_accuracy(y_test_np, y_pred, 50000, 200000)  
    large_acc = fast_sign_accuracy(y_test_np, y_pred, 200000, float('inf'))

    return signs_percent, rmse, mae, r2, mape, small_acc, medium_acc, large_acc


def cross_validate_model(X, y, cv=3):
    """Perform cross-validation with the advanced model (reduced CV folds for speed)"""
    
    # Use a simpler model for CV to save time, but still representative
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation scores with warning suppression
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    
    cv_rmse = np.sqrt(-cv_scores)
    
    return cv_rmse.mean(), cv_rmse.std()


def get_prediction_confidence(model, X, n_estimators_subset=30):
    """Calculate prediction confidence using ensemble variance (optimized for speed)"""
    
    # Get predictions from different models for confidence estimation
    predictions = []
    
    # Suppress warnings during confidence calculation
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        
        # Use available models to estimate confidence
        if hasattr(model, 'models'):
            for name, mdl in model.models.items():
                if name in ['xgb', 'lgb', 'rf']:  # Use fast models only
                    try:
                        pred = mdl.predict(X)
                        predictions.append(pred)
                    except:
                        continue
        
        # If we have at least 2 predictions, calculate variance
        if len(predictions) >= 2:
            predictions_array = np.array(predictions)
            pred_mean = np.mean(predictions_array, axis=0)
            pred_std = np.std(predictions_array, axis=0)
        else:
            # Fallback: use single model prediction with fixed confidence
            pred_mean = model.predict(X)
            pred_std = np.ones_like(pred_mean) * np.std(pred_mean) * 0.1
    
    # Confidence intervals (90% for faster calculation)
    confidence_lower = pred_mean - 1.64 * pred_std
    confidence_upper = pred_mean + 1.64 * pred_std
    
    return pred_std, confidence_lower, confidence_upper