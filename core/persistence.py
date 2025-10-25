"""
Persistence layer for user study results.
"""
import os
import pandas as pd
from datetime import datetime
from typing import Optional
from core.config import config
from core.schema import UserFeedback


def ensure_results_file() -> str:
    """
    Ensure results directory and file exist.
    
    Returns:
        Path to results file
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    results_path = config.get_results_path()
    
    # Create file with headers if doesn't exist
    if not os.path.exists(results_path):
        df = pd.DataFrame(columns=[
            'row_id',
            'anomaly_flag',
            'model_score',
            'llm_short_title',
            'user_is_anom',
            'user_type',
            'confidence',
            'ms_to_decide',
            'timestamp'
        ])
        df.to_csv(results_path, index=False)
    
    return results_path


def log_user_feedback(feedback: UserFeedback) -> bool:
    """
    Append user feedback to results CSV.
    
    Args:
        feedback: UserFeedback object
        
    Returns:
        Success status
    """
    try:
        results_path = ensure_results_file()
        
        # Convert to DataFrame row
        row_data = {
            'row_id': feedback.row_id,
            'anomaly_flag': feedback.anomaly_flag,
            'model_score': feedback.model_score,
            'llm_short_title': feedback.llm_short_title,
            'user_is_anom': feedback.user_is_anom,
            'user_type': feedback.user_type,
            'confidence': feedback.confidence,
            'ms_to_decide': feedback.ms_to_decide,
            'timestamp': feedback.timestamp.isoformat()
        }
        
        df_new = pd.DataFrame([row_data])
        
        # Append to file
        df_new.to_csv(results_path, mode='a', header=False, index=False)
        
        return True
    
    except Exception as e:
        print(f"Failed to log feedback: {e}")
        return False


def load_user_study_results() -> Optional[pd.DataFrame]:
    """
    Load all user study results.
    
    Returns:
        DataFrame of results or None if not exists
    """
    results_path = config.get_results_path()
    
    if not os.path.exists(results_path):
        return None
    
    try:
        df = pd.read_csv(results_path)
        
        if len(df) == 0:
            return None
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    except Exception as e:
        print(f"Failed to load results: {e}")
        return None


def compute_user_study_metrics(results_df: pd.DataFrame) -> dict:
    """
    Compute metrics from user study results.
    
    Args:
        results_df: DataFrame of user feedback
        
    Returns:
        Dict of metrics
    """
    if results_df is None or len(results_df) == 0:
        return {}
    
    metrics = {}
    
    # Basic counts
    metrics['total_reviews'] = len(results_df)
    metrics['flagged_count'] = results_df['anomaly_flag'].sum()
    metrics['user_confirmed_count'] = results_df['user_is_anom'].sum()
    
    # Agreement metrics
    metrics['agreement_rate'] = (
        results_df['anomaly_flag'] == results_df['user_is_anom']
    ).mean()
    
    # Precision: of flagged items, how many were true positives?
    flagged = results_df[results_df['anomaly_flag'] == True]
    if len(flagged) > 0:
        metrics['precision'] = (flagged['user_is_anom'] == True).mean()
    else:
        metrics['precision'] = None
    
    # Recall: of user-confirmed anomalies, how many were flagged?
    user_anoms = results_df[results_df['user_is_anom'] == True]
    if len(user_anoms) > 0:
        metrics['recall'] = (user_anoms['anomaly_flag'] == True).mean()
    else:
        metrics['recall'] = None
    
    # F1 score
    if metrics['precision'] is not None and metrics['recall'] is not None:
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = (
                2 * metrics['precision'] * metrics['recall'] /
                (metrics['precision'] + metrics['recall'])
            )
        else:
            metrics['f1_score'] = 0
    else:
        metrics['f1_score'] = None
    
    # Timing
    metrics['avg_ms_to_decide'] = results_df['ms_to_decide'].mean()
    metrics['median_ms_to_decide'] = results_df['ms_to_decide'].median()
    
    # Confidence
    metrics['avg_confidence'] = results_df['confidence'].mean()
    
    # Anomaly types (user-labeled)
    type_counts = results_df[results_df['user_is_anom'] == True]['user_type'].value_counts()
    metrics['anomaly_type_distribution'] = type_counts.to_dict()
    
    return metrics


def export_results_summary(results_df: pd.DataFrame, output_path: str) -> bool:
    """
    Export enhanced results with metrics to CSV.
    
    Args:
        results_df: User study results
        output_path: Where to save
        
    Returns:
        Success status
    """
    try:
        if results_df is None or len(results_df) == 0:
            return False
        
        # Add derived columns
        df_export = results_df.copy()
        df_export['is_true_positive'] = (
            (df_export['anomaly_flag'] == True) & 
            (df_export['user_is_anom'] == True)
        )
        df_export['is_false_positive'] = (
            (df_export['anomaly_flag'] == True) & 
            (df_export['user_is_anom'] == False)
        )
        df_export['is_false_negative'] = (
            (df_export['anomaly_flag'] == False) & 
            (df_export['user_is_anom'] == True)
        )
        df_export['seconds_to_decide'] = df_export['ms_to_decide'] / 1000
        
        df_export.to_csv(output_path, index=False)
        return True
    
    except Exception as e:
        print(f"Export failed: {e}")
        return False


def clear_user_study_results() -> bool:
    """
    Clear all user study results (for testing/reset).
    
    Returns:
        Success status
    """
    try:
        results_path = config.get_results_path()
        
        if os.path.exists(results_path):
            os.remove(results_path)
        
        # Recreate empty file
        ensure_results_file()
        return True
    
    except Exception as e:
        print(f"Clear failed: {e}")
        return False

