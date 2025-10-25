"""
Rule-based anomaly detection for accounting data.
"""
import pandas as pd
import numpy as np
from typing import Tuple
from core.config import config


def apply_rules(df: pd.DataFrame, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply business rules to detect anomalies.
    
    Args:
        df: Original DataFrame with metadata
        X: Feature matrix
        
    Returns:
        rules_df: DataFrame of boolean flags (one column per rule)
        rule_score: Aggregated rule score per row (0-1)
    """
    rules_df = pd.DataFrame(index=df.index)
    
    # Rule 1: Unbalanced voucher
    rules_df['unbalanced_voucher'] = _is_unbalanced_voucher(df)
    
    # Rule 2: Duplicate document or amount same day
    rules_df['duplicate_same_day'] = _duplicate_doc_or_amount_same_day(df)
    
    # Rule 3: Rare vendor at unusual hour
    rules_df['rare_vendor_unusual_hour'] = _rare_vendor_hour(df, X)
    
    # Rule 4: Round large amount
    rules_df['round_large_amount'] = _round_large_amount(df, X)
    
    # Rule 5: Outside business hours
    rules_df['outside_business_hours'] = _outside_business_hours(df)
    
    # Rule 6: Unusual account pair
    rules_df['unusual_account_pair'] = _unusual_account_pair(df, X)
    
    # Aggregate: normalize by counting triggered rules
    rule_score = rules_df.sum(axis=1) / len(rules_df.columns)
    
    return rules_df, rule_score


def _is_unbalanced_voucher(df: pd.DataFrame) -> pd.Series:
    """Check if voucher debits != credits (tolerance 0.01)."""
    if 'voucher_id' not in df.columns:
        return pd.Series(False, index=df.index)
    
    voucher_balance = df.groupby('voucher_id', group_keys=False).apply(
        lambda g: abs(g['debit'].sum() - g['credit'].sum()) > 0.01,
        include_groups=False
    )
    return df['voucher_id'].map(voucher_balance).fillna(False)


def _duplicate_doc_or_amount_same_day(df: pd.DataFrame) -> pd.Series:
    """Check for duplicate amounts from same vendor on same day."""
    if not all(c in df.columns for c in ['date', 'vendor', 'amount']):
        return pd.Series(False, index=df.index)
    
    # Create composite key
    df_temp = df.copy()
    df_temp['dup_key'] = (
        df_temp['date'].astype(str) + '_' + 
        df_temp['vendor'].astype(str) + '_' + 
        df_temp['amount'].abs().round(2).astype(str)
    )
    
    dup_counts = df_temp['dup_key'].value_counts()
    return (df_temp['dup_key'].map(dup_counts) > 1).fillna(False)


def _rare_vendor_hour(df: pd.DataFrame, X: pd.DataFrame) -> pd.Series:
    """Rare vendor (appears <= 3 times) posting at unusual hour."""
    if 'vendor_freq' not in X.columns or 'hour' not in X.columns:
        return pd.Series(False, index=df.index)
    
    rare_vendor = X['vendor_freq'] <= config.MIN_VENDOR_FREQ
    unusual_hour = (X['hour'] < 6) | (X['hour'] > 22)
    
    return rare_vendor & unusual_hour


def _round_large_amount(df: pd.DataFrame, X: pd.DataFrame) -> pd.Series:
    """Large round amounts (>= $1000, ends in 00)."""
    if 'amount_abs' not in X.columns or 'is_round_amount' not in X.columns:
        return pd.Series(False, index=df.index)
    
    return (X['amount_abs'] >= 1000) & (X['is_round_amount'] == 1)


def _outside_business_hours(df: pd.DataFrame) -> pd.Series:
    """Posted outside standard business hours (7am-7pm)."""
    if 'date_parsed' not in df.columns:
        return pd.Series(False, index=df.index)
    
    hour = df['date_parsed'].dt.hour
    return ((hour < config.BUSINESS_HOURS_START) | (hour >= config.BUSINESS_HOURS_END)).fillna(False)


def _unusual_account_pair(df: pd.DataFrame, X: pd.DataFrame) -> pd.Series:
    """Account pair rarely seen together (frequency <= 2)."""
    if 'account_pair_freq' not in X.columns:
        return pd.Series(False, index=df.index)
    
    return X['account_pair_freq'] <= 2


def get_triggered_rules(rules_df: pd.DataFrame, idx: int) -> list:
    """Get list of rule names triggered for a specific row."""
    if idx not in rules_df.index:
        return []
    
    row = rules_df.loc[idx]
    return [col for col, val in row.items() if val]


def format_rules_summary(rules_df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics for rules."""
    summary = pd.DataFrame({
        'Rule': rules_df.columns,
        'Triggered': rules_df.sum().values,
        'Percentage': (rules_df.sum() / len(rules_df) * 100).round(2).values
    })
    return summary.sort_values('Triggered', ascending=False)

