"""
Feature engineering for accounting ledger data.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List
from datetime import datetime


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], pd.DataFrame, pd.DataFrame]:
    """
    Build feature matrix from raw ledger data.
    
    Args:
        df: Raw ledger DataFrame
        
    Returns:
        X: Feature matrix (numeric, ready for ML)
        feature_cols: List of feature column names
        meta: Metadata DataFrame (original fields + derived but non-numeric)
        flags_df: Boolean flags from rule checks
    """
    df = df.copy()
    
    # === 1. Clean and normalize ===
    df = _clean_data(df)
    
    # === 2. Derive temporal features ===
    df = _add_temporal_features(df)
    
    # === 3. Derive amount-based features ===
    df = _add_amount_features(df)
    
    # === 4. Derive entity frequency features ===
    df = _add_frequency_features(df)
    
    # === 5. Derive transaction pattern features ===
    df = _add_pattern_features(df)
    
    # === 6. Select numeric features for X ===
    feature_cols = [
        'amount_log',
        'amount_z',
        'vendor_freq',
        'poster_freq',
        'weekday',
        'hour',
        'is_month_end',
        'account_pair_freq',
        'duplicate_amount_same_day',
        'is_balanced_voucher',
        'is_round_amount',
        'amount_abs',
        'debit_credit_ratio'
    ]
    
    # Handle missing values in features
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
    
    X = df[feature_cols].copy()
    
    # === 7. Meta columns (for display, not training) ===
    meta_cols = ['date', 'voucher_id', 'account', 'amount', 'vendor', 'poster', 'description']
    meta_cols = [c for c in meta_cols if c in df.columns]
    meta = df[meta_cols].copy()
    
    # Add derived metadata
    if 'date_parsed' in df.columns:
        meta['date_parsed'] = df['date_parsed']
    
    # === 8. Rule flags (computed separately in rules.py but we signal them here) ===
    flags_df = pd.DataFrame(index=df.index)
    flags_df['duplicate_amount_same_day'] = df['duplicate_amount_same_day'] > 0
    flags_df['is_balanced_voucher'] = df['is_balanced_voucher'] == 1
    flags_df['is_round_amount'] = df['is_round_amount'] == 1
    
    return X, feature_cols, meta, flags_df


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize basic fields."""
    df = df.copy()
    
    # Parse dates
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Ensure amount is numeric
    if 'amount' not in df.columns:
        if 'debit' in df.columns and 'credit' in df.columns:
            df['debit'] = pd.to_numeric(df['debit'], errors='coerce').fillna(0)
            df['credit'] = pd.to_numeric(df['credit'], errors='coerce').fillna(0)
            df['amount'] = df['debit'] + df['credit']
        else:
            df['amount'] = 0
    
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    df['amount_abs'] = df['amount'].abs()
    
    # Handle debit/credit if they exist
    if 'debit' in df.columns:
        df['debit'] = pd.to_numeric(df['debit'], errors='coerce').fillna(0)
    else:
        df['debit'] = df['amount'].clip(lower=0)
    
    if 'credit' in df.columns:
        df['credit'] = pd.to_numeric(df['credit'], errors='coerce').fillna(0)
    else:
        df['credit'] = (-df['amount']).clip(lower=0)
    
    # Debit/credit ratio
    df['debit_credit_ratio'] = np.where(
        df['credit'] > 0,
        df['debit'] / (df['credit'] + 1e-6),
        0
    )
    
    # Normalize text fields
    for col in ['vendor', 'poster', 'account', 'voucher_id']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    
    if 'description' in df.columns:
        df['description'] = df['description'].fillna('').astype(str)
    
    return df


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    df = df.copy()
    
    if 'date_parsed' in df.columns:
        df['weekday'] = df['date_parsed'].dt.weekday  # Monday=0, Sunday=6
        df['hour'] = df['date_parsed'].dt.hour
        df['day_of_month'] = df['date_parsed'].dt.day
        df['month'] = df['date_parsed'].dt.month
        
        # Month-end flag (last 3 days of month)
        df['is_month_end'] = (df['date_parsed'].dt.days_in_month - df['day_of_month'] <= 2).astype(int)
    else:
        df['weekday'] = 2  # Default to Wednesday
        df['hour'] = 12  # Default to noon
        df['is_month_end'] = 0
    
    return df


def _add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add amount-based statistical features."""
    df = df.copy()
    
    # Log-transform (for skewed distributions)
    df['amount_log'] = np.log1p(df['amount_abs'])
    
    # Z-score within account (how unusual is this amount for this account?)
    if 'account' in df.columns:
        df['amount_z'] = df.groupby('account')['amount_abs'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6) if len(x) > 1 else 0
        )
    else:
        df['amount_z'] = 0
    
    # Round amount detection (ends in 00, 000, etc.)
    df['is_round_amount'] = (
        (df['amount_abs'] % 100 == 0) & (df['amount_abs'] >= 100)
    ).astype(int)
    
    return df


def _add_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add entity frequency features."""
    df = df.copy()
    
    # Vendor frequency
    if 'vendor' in df.columns:
        vendor_counts = df['vendor'].value_counts()
        df['vendor_freq'] = df['vendor'].map(vendor_counts).fillna(1)
    else:
        df['vendor_freq'] = 1
    
    # Poster frequency
    if 'poster' in df.columns:
        poster_counts = df['poster'].value_counts()
        df['poster_freq'] = df['poster'].map(poster_counts).fillna(1)
    else:
        df['poster_freq'] = 1
    
    # Account pair frequency (if we have voucher grouping)
    if 'voucher_id' in df.columns and 'account' in df.columns:
        # Count how often this account appears with others in same voucher
        voucher_accounts = df.groupby('voucher_id')['account'].apply(lambda x: tuple(sorted(x.unique())))
        account_pair_counts = voucher_accounts.value_counts()
        df['account_pair_freq'] = df['voucher_id'].map(
            voucher_accounts.map(account_pair_counts)
        ).fillna(1)
    else:
        df['account_pair_freq'] = 1
    
    return df


def _add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add transaction pattern features."""
    df = df.copy()
    
    # Duplicate amount on same day (same vendor or not)
    if 'date' in df.columns and 'vendor' in df.columns:
        dup_key = df['date'].astype(str) + '_' + df['vendor'].astype(str) + '_' + df['amount_abs'].astype(str)
        dup_counts = dup_key.value_counts()
        df['duplicate_amount_same_day'] = dup_key.map(dup_counts) - 1  # Subtract self
    else:
        df['duplicate_amount_same_day'] = 0
    
    # Balanced voucher check (sum of debits = sum of credits per voucher)
    if 'voucher_id' in df.columns:
        voucher_balance = df.groupby('voucher_id', group_keys=False).apply(
            lambda g: abs(g['debit'].sum() - g['credit'].sum()) < 0.01,
            include_groups=False
        )
        df['is_balanced_voucher'] = df['voucher_id'].map(voucher_balance).astype(int)
    else:
        df['is_balanced_voucher'] = 1  # Assume balanced if no voucher info
    
    return df

