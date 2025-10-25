"""
Machine learning-based anomaly detection.
Combines IsolationForest, XGBoost, and rule-based scores.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Tuple, Optional
from core.config import config

# Conditional XGBoost import
if config.ENABLE_XGBOOST:
    try:
        import xgboost as xgb
    except ImportError:
        config.ENABLE_XGBOOST = False


def fit_iforest(X: pd.DataFrame) -> IsolationForest:
    """
    Fit IsolationForest on feature matrix.
    
    Args:
        X: Feature matrix
        
    Returns:
        Trained IsolationForest model
    """
    model = IsolationForest(
        contamination=config.IFOREST_CONTAMINATION,
        n_estimators=config.IFOREST_N_ESTIMATORS,
        max_samples=min(config.IFOREST_MAX_SAMPLES, len(X)),
        random_state=config.IFOREST_RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X)
    return model


def score_iforest(model: IsolationForest, X: pd.DataFrame) -> np.ndarray:
    """
    Score samples with IsolationForest, normalized to [0, 1].
    
    Args:
        model: Trained IsolationForest
        X: Feature matrix
        
    Returns:
        Anomaly scores (0 = normal, 1 = most anomalous)
    """
    # decision_function returns negative scores (more negative = more anomalous)
    raw_scores = model.decision_function(X)
    
    # Normalize to [0, 1]: more negative -> higher score
    scores = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-6)
    
    return scores


def fit_xgboost_typicality(
    X: pd.DataFrame,
    target_col: str = 'amount_abs'
) -> Optional[object]:
    """
    Train XGBoost to predict 'typical' amounts.
    Residuals indicate atypicality.
    
    Args:
        X: Feature matrix
        target_col: Column to predict
        
    Returns:
        Trained XGBoost model or None if disabled
    """
    if not config.ENABLE_XGBOOST:
        return None
    
    if target_col not in X.columns:
        return None
    
    # Prepare data
    y = X[target_col].values
    X_train = X.drop(columns=[target_col], errors='ignore')
    
    # Train
    model = xgb.XGBRegressor(
        n_estimators=config.XGBOOST_N_ESTIMATORS,
        max_depth=config.XGBOOST_MAX_DEPTH,
        learning_rate=config.XGBOOST_LEARNING_RATE,
        random_state=config.XGBOOST_RANDOM_STATE,
        objective='reg:squarederror',
        n_jobs=-1
    )
    
    model.fit(X_train, y)
    return model


def score_xgboost_typicality(
    model: Optional[object],
    X: pd.DataFrame,
    target_col: str = 'amount_abs'
) -> np.ndarray:
    """
    Score based on prediction residuals (normalized to [0, 1]).
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix
        target_col: Target column name
        
    Returns:
        Anomaly scores based on residuals
    """
    if model is None or not config.ENABLE_XGBOOST:
        return np.zeros(len(X))
    
    if target_col not in X.columns:
        return np.zeros(len(X))
    
    y_true = X[target_col].values
    X_pred = X.drop(columns=[target_col], errors='ignore')
    y_pred = model.predict(X_pred)
    
    # Compute absolute residuals
    residuals = np.abs(y_true - y_pred)
    
    # Normalize using robust scaling (percentile-based)
    p95 = np.percentile(residuals, 95)
    scores = np.clip(residuals / (p95 + 1e-6), 0, 1)
    
    return scores


def ensemble_scores(
    iso_scores: np.ndarray,
    xgb_scores: np.ndarray,
    rule_scores: np.ndarray
) -> np.ndarray:
    """
    Combine scores from different detectors.
    
    Args:
        iso_scores: IsolationForest scores
        xgb_scores: XGBoost residual scores
        rule_scores: Rule-based scores
        
    Returns:
        Combined anomaly scores (weighted average)
    """
    # Normalize weights
    total_weight = config.WEIGHT_IFOREST + config.WEIGHT_XGBOOST + config.WEIGHT_RULES
    w_iso = config.WEIGHT_IFOREST / total_weight
    w_xgb = config.WEIGHT_XGBOOST / total_weight
    w_rule = config.WEIGHT_RULES / total_weight
    
    combined = w_iso * iso_scores + w_xgb * xgb_scores + w_rule * rule_scores
    
    return combined


def compute_threshold(scores: np.ndarray, k: float = None) -> float:
    """
    Compute robust threshold using Median Absolute Deviation (MAD).
    
    Args:
        scores: Anomaly scores
        k: Multiplier for MAD (default from config)
        
    Returns:
        Threshold value
    """
    if k is None:
        k = config.THRESHOLD_K
    
    median = np.median(scores)
    mad = np.median(np.abs(scores - median))
    
    # MAD-based threshold (robust to outliers)
    threshold = median + k * mad
    
    return threshold


def flag_anomalies(
    scores: np.ndarray,
    rules_df: pd.DataFrame,
    threshold: float
) -> pd.Series:
    """
    Flag anomalies based on score threshold and critical rules.
    
    Args:
        scores: Combined anomaly scores
        rules_df: DataFrame of rule flags
        threshold: Score threshold
        
    Returns:
        Boolean series of anomaly flags
    """
    # Critical rules that auto-flag regardless of score
    critical_rules = ['unbalanced_voucher']
    has_critical = rules_df[
        [c for c in critical_rules if c in rules_df.columns]
    ].any(axis=1)
    
    # Flag if score >= threshold OR critical rule triggered
    flags = (scores >= threshold) | has_critical
    
    return flags


class AnomalyDetector:
    """
    Unified anomaly detector combining multiple methods.
    """
    
    def __init__(self):
        self.iforest_model = None
        self.xgboost_model = None
        self.threshold = None
        self.feature_cols = None
    
    def fit(self, X: pd.DataFrame, feature_cols: list):
        """Fit all models."""
        self.feature_cols = feature_cols
        
        # Fit IsolationForest
        self.iforest_model = fit_iforest(X)
        
        # Fit XGBoost (optional)
        if config.ENABLE_XGBOOST:
            self.xgboost_model = fit_xgboost_typicality(X)
    
    def predict(
        self,
        X: pd.DataFrame,
        rule_scores: np.ndarray,
        rules_df: pd.DataFrame,
        k: float = None
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Predict anomaly scores and flags.
        
        Returns:
            scores: Combined anomaly scores
            flags: Boolean anomaly flags
            threshold: Computed threshold
        """
        # Score with IsolationForest
        iso_scores = score_iforest(self.iforest_model, X)
        
        # Score with XGBoost
        xgb_scores = score_xgboost_typicality(self.xgboost_model, X)
        
        # Ensemble
        scores = ensemble_scores(iso_scores, xgb_scores, rule_scores)
        
        # Compute threshold
        threshold = compute_threshold(scores, k)
        self.threshold = threshold
        
        # Flag anomalies
        flags = flag_anomalies(scores, rules_df, threshold)
        
        return scores, flags, threshold

