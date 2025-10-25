"""
Tests for anomaly detection models.
"""
import pytest
import pandas as pd
import numpy as np
from core.detectors import (
    fit_iforest,
    score_iforest,
    ensemble_scores,
    compute_threshold,
    flag_anomalies,
    AnomalyDetector
)


def create_sample_features():
    """Create sample feature matrix."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.randn(50),
        'feature2': np.random.randn(50),
        'feature3': np.random.randn(50) * 2
    })


def test_fit_iforest():
    """Test IsolationForest fitting."""
    X = create_sample_features()
    model = fit_iforest(X)
    
    assert model is not None, "Model should be fitted"
    assert hasattr(model, 'decision_function'), "Model should have decision_function"


def test_score_iforest_range():
    """Test that IsolationForest scores are in [0, 1]."""
    X = create_sample_features()
    model = fit_iforest(X)
    scores = score_iforest(model, X)
    
    assert len(scores) == len(X), "Should have one score per row"
    assert (scores >= 0).all(), "Scores should be >= 0"
    assert (scores <= 1).all(), "Scores should be <= 1"


def test_score_iforest_finds_outliers():
    """Test that extreme outliers get high scores."""
    X = create_sample_features()
    
    # Add obvious outlier
    outlier_row = pd.DataFrame({
        'feature1': [100],
        'feature2': [100],
        'feature3': [100]
    })
    X_with_outlier = pd.concat([X, outlier_row], ignore_index=True)
    
    model = fit_iforest(X_with_outlier)
    scores = score_iforest(model, X_with_outlier)
    
    # Outlier should have high score
    outlier_score = scores[-1]
    median_score = np.median(scores[:-1])
    
    assert outlier_score > median_score, "Outlier should score higher than median"


def test_ensemble_scores():
    """Test ensemble score combination."""
    iso_scores = np.array([0.5, 0.8, 0.3])
    xgb_scores = np.array([0.4, 0.9, 0.2])
    rule_scores = np.array([0.6, 0.7, 0.1])
    
    combined = ensemble_scores(iso_scores, xgb_scores, rule_scores)
    
    assert len(combined) == len(iso_scores), "Should have same length"
    assert (combined >= 0).all(), "Scores should be >= 0"
    assert (combined <= 1).all(), "Scores should be <= 1"


def test_ensemble_scores_weighted():
    """Test that ensemble properly weights components."""
    # All zeros except one component
    iso_scores = np.array([1.0, 0.0, 0.0])
    xgb_scores = np.array([0.0, 1.0, 0.0])
    rule_scores = np.array([0.0, 0.0, 1.0])
    
    combined = ensemble_scores(iso_scores, xgb_scores, rule_scores)
    
    # All should be between 0 and 1
    assert all(0 < combined) and all(combined < 1), "Weighted combination should be partial"


def test_compute_threshold():
    """Test MAD-based threshold computation."""
    scores = np.array([0.1, 0.2, 0.15, 0.18, 0.22, 0.9, 0.95])
    
    threshold = compute_threshold(scores, k=2.0)
    
    assert threshold > np.median(scores), "Threshold should be above median"
    assert threshold < np.max(scores), "Threshold should be below max (usually)"


def test_compute_threshold_sensitivity():
    """Test that higher k gives higher threshold."""
    scores = np.array([0.1, 0.2, 0.15, 0.18, 0.22, 0.9, 0.95])
    
    threshold_low = compute_threshold(scores, k=1.5)
    threshold_high = compute_threshold(scores, k=3.0)
    
    assert threshold_high > threshold_low, "Higher k should give higher threshold"


def test_flag_anomalies():
    """Test anomaly flagging logic."""
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = 0.5
    
    # Create dummy rules DataFrame
    rules_df = pd.DataFrame({
        'unbalanced_voucher': [False, False, False, True],
        'other_rule': [False, False, True, False]
    })
    
    flags = flag_anomalies(scores, rules_df, threshold)
    
    assert len(flags) == len(scores), "Should have one flag per row"
    assert flags[2] == True, "Score 0.8 > 0.5 should be flagged"
    assert flags[3] == True, "Critical rule should flag regardless of score"


def test_flag_anomalies_critical_rules():
    """Test that critical rules override score threshold."""
    scores = np.array([0.1, 0.2])  # Low scores
    threshold = 0.5
    
    rules_df = pd.DataFrame({
        'unbalanced_voucher': [True, False]
    })
    
    flags = flag_anomalies(scores, rules_df, threshold)
    
    assert flags[0] == True, "Critical rule should flag even with low score"
    assert flags[1] == False, "Low score without critical rule should not flag"


def test_anomaly_detector_fit():
    """Test AnomalyDetector fit method."""
    X = create_sample_features()
    detector = AnomalyDetector()
    
    feature_cols = X.columns.tolist()
    detector.fit(X, feature_cols)
    
    assert detector.iforest_model is not None, "IsolationForest should be fitted"
    assert detector.feature_cols == feature_cols, "Feature columns should be stored"


def test_anomaly_detector_predict():
    """Test AnomalyDetector predict method."""
    X = create_sample_features()
    detector = AnomalyDetector()
    
    feature_cols = X.columns.tolist()
    detector.fit(X, feature_cols)
    
    rule_scores = np.random.rand(len(X))
    rules_df = pd.DataFrame({
        'rule1': np.random.choice([True, False], len(X)),
        'rule2': np.random.choice([True, False], len(X))
    })
    
    scores, flags, threshold = detector.predict(X, rule_scores, rules_df)
    
    assert len(scores) == len(X), "Should have one score per row"
    assert len(flags) == len(X), "Should have one flag per row"
    assert threshold > 0, "Threshold should be positive"
    assert (scores >= 0).all() and (scores <= 1).all(), "Scores should be in [0,1]"


def test_anomaly_detector_consistency():
    """Test that detector gives consistent results with same data."""
    X = create_sample_features()
    
    detector1 = AnomalyDetector()
    detector1.fit(X, X.columns.tolist())
    
    rule_scores = np.zeros(len(X))
    rules_df = pd.DataFrame({'rule1': [False] * len(X)})
    
    scores1, flags1, threshold1 = detector1.predict(X, rule_scores, rules_df)
    scores2, flags2, threshold2 = detector1.predict(X, rule_scores, rules_df)
    
    # Should be identical on same data
    np.testing.assert_array_almost_equal(scores1, scores2, decimal=5)
    assert (flags1 == flags2).all(), "Flags should be identical"
    assert threshold1 == threshold2, "Threshold should be identical"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

