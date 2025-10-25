"""
SHAP-based explainability for XGBoost models.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from core.config import config

# Conditional imports
if config.ENABLE_SHAP:
    try:
        import shap
    except ImportError:
        config.ENABLE_SHAP = False


def compute_shap_values(
    model: Optional[object],
    X: pd.DataFrame,
    background_samples: int = 100
) -> Optional[object]:
    """
    Compute SHAP values for XGBoost model.
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix
        background_samples: Number of samples for background
        
    Returns:
        SHAP Explanation object or None
    """
    if model is None or not config.ENABLE_SHAP:
        return None
    
    try:
        # Remove target column if present
        X_explain = X.drop(columns=['amount_abs'], errors='ignore')
        
        # Create explainer (TreeExplainer is fast for XGBoost)
        explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values
        shap_values = explainer(X_explain)
        
        return shap_values
    except Exception as e:
        print(f"SHAP computation failed: {e}")
        return None


def get_top_shap_features(
    shap_values: Optional[object],
    X: pd.DataFrame,
    row_idx: int,
    top_k: int = None
) -> List[Dict]:
    """
    Get top-k SHAP features for a specific row.
    
    Args:
        shap_values: SHAP Explanation object
        X: Feature matrix
        row_idx: Row index
        top_k: Number of top features
        
    Returns:
        List of dicts with feature name, value, and SHAP contribution
    """
    if shap_values is None or not config.ENABLE_SHAP:
        return []
    
    if top_k is None:
        top_k = config.SHAP_TOP_K
    
    try:
        # Remove target column if present
        X_explain = X.drop(columns=['amount_abs'], errors='ignore')
        
        # Get SHAP values for this row
        row_shap = shap_values[row_idx].values
        row_features = X_explain.iloc[row_idx]
        feature_names = X_explain.columns
        
        # Sort by absolute SHAP value
        importance_order = np.argsort(np.abs(row_shap))[::-1]
        
        # Build result
        top_features = []
        for idx in importance_order[:top_k]:
            top_features.append({
                'feature': feature_names[idx],
                'value': float(row_features.iloc[idx]),
                'shap_contribution': float(row_shap[idx])
            })
        
        return top_features
    except Exception as e:
        print(f"SHAP feature extraction failed: {e}")
        return []


def format_shap_for_display(top_features: List[Dict]) -> pd.DataFrame:
    """Format SHAP features as a DataFrame for display."""
    if not top_features:
        return pd.DataFrame()
    
    df = pd.DataFrame(top_features)
    df['abs_contribution'] = df['shap_contribution'].abs()
    df = df.sort_values('abs_contribution', ascending=False)
    df['shap_contribution'] = df['shap_contribution'].round(3)
    df['value'] = df['value'].round(3)
    
    return df[['feature', 'value', 'shap_contribution']]


def get_shap_summary(
    shap_values: Optional[object],
    X: pd.DataFrame,
    max_display: int = None
) -> Optional[pd.DataFrame]:
    """
    Get global feature importance summary from SHAP.
    
    Args:
        shap_values: SHAP Explanation object
        X: Feature matrix
        max_display: Max features to display
        
    Returns:
        DataFrame with feature importance
    """
    if shap_values is None or not config.ENABLE_SHAP:
        return None
    
    if max_display is None:
        max_display = config.SHAP_MAX_DISPLAY
    
    try:
        X_explain = X.drop(columns=['amount_abs'], errors='ignore')
        
        # Compute mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        
        summary = pd.DataFrame({
            'feature': X_explain.columns,
            'mean_abs_shap': mean_abs_shap
        })
        
        summary = summary.sort_values('mean_abs_shap', ascending=False).head(max_display)
        summary['mean_abs_shap'] = summary['mean_abs_shap'].round(4)
        
        return summary
    except Exception as e:
        print(f"SHAP summary failed: {e}")
        return None


def create_shap_waterfall_data(
    shap_values: Optional[object],
    X: pd.DataFrame,
    row_idx: int,
    max_display: int = 10
) -> Optional[Dict]:
    """
    Create data for waterfall chart visualization.
    
    Args:
        shap_values: SHAP Explanation object
        X: Feature matrix
        row_idx: Row index
        max_display: Max features to show
        
    Returns:
        Dict with data for plotting
    """
    if shap_values is None or not config.ENABLE_SHAP:
        return None
    
    try:
        X_explain = X.drop(columns=['amount_abs'], errors='ignore')
        
        row_shap = shap_values[row_idx].values
        row_features = X_explain.iloc[row_idx]
        feature_names = X_explain.columns
        
        # Sort by absolute contribution
        importance_order = np.argsort(np.abs(row_shap))[::-1][:max_display]
        
        return {
            'features': [feature_names[i] for i in importance_order],
            'values': [float(row_features.iloc[i]) for i in importance_order],
            'shap_values': [float(row_shap[i]) for i in importance_order],
            'base_value': float(shap_values[row_idx].base_values)
        }
    except Exception as e:
        print(f"Waterfall data creation failed: {e}")
        return None

