"""
Tests for feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np
from core.features import build_features


def create_sample_df():
    """Create a minimal sample DataFrame for testing."""
    return pd.DataFrame({
        'date': ['2024-10-01 09:00:00', '2024-10-01 10:00:00', '2024-10-02 09:00:00'],
        'voucher_id': ['V001', 'V001', 'V002'],
        'account': ['1000-CASH', '5000-EXPENSE', '1000-CASH'],
        'debit': [1000.0, 0.0, 500.0],
        'credit': [0.0, 1000.0, 0.0],
        'amount': [1000.0, -1000.0, 500.0],
        'vendor': ['Vendor A', 'Vendor A', 'Vendor B'],
        'poster': ['john.doe', 'john.doe', 'jane.smith'],
        'description': ['Test 1', 'Test 2', 'Test 3']
    })


def test_build_features_returns_correct_shapes():
    """Test that build_features returns correct output shapes."""
    df = create_sample_df()
    X, feature_cols, meta, flags_df = build_features(df)
    
    # Check shapes match
    assert len(X) == len(df), "Feature matrix should have same length as input"
    assert len(meta) == len(df), "Metadata should have same length as input"
    assert len(flags_df) == len(df), "Flags should have same length as input"
    
    # Check feature columns are returned
    assert len(feature_cols) > 0, "Should have at least some features"
    assert all(col in X.columns for col in feature_cols), "All feature cols should be in X"


def test_build_features_handles_missing_values():
    """Test that missing values are handled correctly."""
    df = create_sample_df()
    df.loc[0, 'vendor'] = None
    df.loc[1, 'poster'] = None
    
    X, feature_cols, meta, flags_df = build_features(df)
    
    # Check no NaN in feature matrix
    assert not X.isnull().any().any(), "Feature matrix should not contain NaN"


def test_build_features_creates_expected_features():
    """Test that expected features are created."""
    df = create_sample_df()
    X, feature_cols, meta, flags_df = build_features(df)
    
    expected_features = [
        'amount_log',
        'amount_z',
        'vendor_freq',
        'poster_freq',
        'weekday',
        'hour'
    ]
    
    for feat in expected_features:
        assert feat in feature_cols, f"Feature {feat} should be present"
        assert feat in X.columns, f"Feature {feat} should be in X"


def test_build_features_numeric_types():
    """Test that all features are numeric."""
    df = create_sample_df()
    X, feature_cols, meta, flags_df = build_features(df)
    
    for col in feature_cols:
        assert pd.api.types.is_numeric_dtype(X[col]), f"Feature {col} should be numeric"


def test_temporal_features():
    """Test temporal feature extraction."""
    df = pd.DataFrame({
        'date': ['2024-10-15 14:30:00', '2024-10-31 23:59:00'],
        'voucher_id': ['V001', 'V002'],
        'account': ['1000-CASH', '1000-CASH'],
        'debit': [100.0, 200.0],
        'credit': [0.0, 0.0],
        'amount': [100.0, 200.0],
        'vendor': ['A', 'B'],
        'poster': ['john', 'jane'],
        'description': ['Test', 'Test']
    })
    
    X, feature_cols, meta, flags_df = build_features(df)
    
    # Check weekday exists and is reasonable
    assert 'weekday' in X.columns
    assert X['weekday'].min() >= 0
    assert X['weekday'].max() <= 6
    
    # Check hour exists
    assert 'hour' in X.columns
    assert X['hour'].min() >= 0
    assert X['hour'].max() <= 23
    
    # Check month-end flag
    assert 'is_month_end' in X.columns
    assert X.loc[1, 'is_month_end'] == 1, "Oct 31 should be flagged as month-end"


def test_amount_features():
    """Test amount-based features."""
    df = create_sample_df()
    X, feature_cols, meta, flags_df = build_features(df)
    
    # Check amount_log is positive for positive amounts
    assert 'amount_log' in X.columns
    assert (X['amount_log'] >= 0).all(), "Log amounts should be non-negative"
    
    # Check amount_z exists
    assert 'amount_z' in X.columns


def test_frequency_features():
    """Test frequency-based features."""
    df = create_sample_df()
    X, feature_cols, meta, flags_df = build_features(df)
    
    # Vendor frequency
    assert 'vendor_freq' in X.columns
    assert X.loc[0, 'vendor_freq'] == 2, "Vendor A appears twice"
    assert X.loc[2, 'vendor_freq'] == 1, "Vendor B appears once"
    
    # Poster frequency
    assert 'poster_freq' in X.columns


def test_balanced_voucher_feature():
    """Test balanced voucher detection."""
    df = create_sample_df()
    X, feature_cols, meta, flags_df = build_features(df)
    
    assert 'is_balanced_voucher' in X.columns
    
    # V001 should be balanced (debit=credit=1000)
    assert X.loc[0, 'is_balanced_voucher'] == 1
    assert X.loc[1, 'is_balanced_voucher'] == 1


def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    df = pd.DataFrame(columns=['date', 'voucher_id', 'account', 'amount'])
    
    X, feature_cols, meta, flags_df = build_features(df)
    
    assert len(X) == 0, "Empty input should produce empty output"
    assert len(feature_cols) > 0, "Feature columns should still be defined"


def test_single_row_dataframe():
    """Test handling of single-row DataFrame."""
    df = create_sample_df().iloc[:1]
    
    X, feature_cols, meta, flags_df = build_features(df)
    
    assert len(X) == 1, "Single row should produce single row output"
    assert not X.isnull().any().any(), "No NaN values should be present"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

