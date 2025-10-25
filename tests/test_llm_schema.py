"""
Tests for LLM schema and explanation validation.
"""
import pytest
import json
from pydantic import ValidationError
from core.schema import LLMExplanation, UserFeedback, LedgerRow
from datetime import datetime


def test_llm_explanation_valid():
    """Test that valid LLM explanation passes validation."""
    data = {
        "short_title": "Test anomaly",
        "rationale": "This is a test rationale explaining the anomaly.",
        "fields_referenced": ["amount", "vendor"],
        "risk_level": "medium",
        "suggested_action": "Verify with vendor",
        "anomaly_type": "duplicate",
        "provenance": {"model_score": 0.85}
    }
    
    explanation = LLMExplanation(**data)
    
    assert explanation.short_title == "Test anomaly"
    assert explanation.risk_level == "medium"
    assert explanation.anomaly_type == "duplicate"


def test_llm_explanation_invalid_risk_level():
    """Test that invalid risk level raises error."""
    data = {
        "short_title": "Test",
        "rationale": "Test rationale",
        "fields_referenced": ["field1"],
        "risk_level": "invalid",  # Invalid
        "suggested_action": "Do something",
        "anomaly_type": "duplicate"
    }
    
    with pytest.raises(ValidationError):
        LLMExplanation(**data)


def test_llm_explanation_invalid_anomaly_type():
    """Test that invalid anomaly type raises error."""
    data = {
        "short_title": "Test",
        "rationale": "Test rationale",
        "fields_referenced": ["field1"],
        "risk_level": "low",
        "suggested_action": "Do something",
        "anomaly_type": "invalid_type"  # Invalid
    }
    
    with pytest.raises(ValidationError):
        LLMExplanation(**data)


def test_llm_explanation_all_risk_levels():
    """Test all valid risk levels."""
    for risk_level in ["low", "medium", "high"]:
        data = {
            "short_title": "Test",
            "rationale": "Test",
            "fields_referenced": ["field1"],
            "risk_level": risk_level,
            "suggested_action": "Test",
            "anomaly_type": "other"
        }
        explanation = LLMExplanation(**data)
        assert explanation.risk_level == risk_level


def test_llm_explanation_all_anomaly_types():
    """Test all valid anomaly types."""
    valid_types = ["duplicate", "misclassification", "unbalanced", "timing_policy", "other"]
    
    for anom_type in valid_types:
        data = {
            "short_title": "Test",
            "rationale": "Test",
            "fields_referenced": ["field1"],
            "risk_level": "low",
            "suggested_action": "Test",
            "anomaly_type": anom_type
        }
        explanation = LLMExplanation(**data)
        assert explanation.anomaly_type == anom_type


def test_llm_explanation_json_serialization():
    """Test that explanation can be serialized to JSON."""
    data = {
        "short_title": "Test",
        "rationale": "Test rationale",
        "fields_referenced": ["amount", "vendor"],
        "risk_level": "high",
        "suggested_action": "Review immediately",
        "anomaly_type": "unbalanced",
        "provenance": {"score": 0.95, "rules": ["rule1"]}
    }
    
    explanation = LLMExplanation(**data)
    json_str = explanation.model_dump_json()
    parsed = json.loads(json_str)
    
    assert parsed["short_title"] == "Test"
    assert parsed["risk_level"] == "high"


def test_user_feedback_valid():
    """Test valid user feedback."""
    feedback = UserFeedback(
        row_id=42,
        anomaly_flag=True,
        model_score=0.87,
        llm_short_title="Test anomaly",
        user_is_anom=True,
        user_type="duplicate",
        confidence=6,
        ms_to_decide=5000
    )
    
    assert feedback.row_id == 42
    assert feedback.confidence == 6
    assert feedback.user_type == "duplicate"


def test_user_feedback_confidence_bounds():
    """Test that confidence is bounded [1, 7]."""
    # Valid
    feedback = UserFeedback(
        row_id=1, anomaly_flag=True, model_score=0.5,
        user_is_anom=True, confidence=7, ms_to_decide=1000
    )
    assert feedback.confidence == 7
    
    # Too high
    with pytest.raises(ValidationError):
        UserFeedback(
            row_id=1, anomaly_flag=True, model_score=0.5,
            user_is_anom=True, confidence=8, ms_to_decide=1000
        )
    
    # Too low
    with pytest.raises(ValidationError):
        UserFeedback(
            row_id=1, anomaly_flag=True, model_score=0.5,
            user_is_anom=True, confidence=0, ms_to_decide=1000
        )


def test_user_feedback_timestamp_auto():
    """Test that timestamp is auto-generated."""
    before = datetime.now()
    
    feedback = UserFeedback(
        row_id=1, anomaly_flag=True, model_score=0.5,
        user_is_anom=True, confidence=5, ms_to_decide=1000
    )
    
    after = datetime.now()
    
    assert before <= feedback.timestamp <= after


def test_ledger_row_valid():
    """Test valid ledger row."""
    row = LedgerRow(
        date="2024-10-23",
        voucher_id="V001",
        account="1000-CASH",
        debit=1000.0,
        credit=0.0,
        amount=1000.0,
        vendor="Test Vendor",
        poster="john.doe",
        description="Test transaction"
    )
    
    assert row.voucher_id == "V001"
    assert row.amount == 1000.0


def test_ledger_row_optional_fields():
    """Test that optional fields can be None."""
    row = LedgerRow(
        date="2024-10-23",
        voucher_id="V001",
        account="1000-CASH",
        amount=1000.0
    )
    
    assert row.debit is None
    assert row.credit is None
    assert row.vendor is None


def test_llm_explanation_field_lengths():
    """Test field length constraints."""
    # Title too long (>100 chars)
    with pytest.raises(ValidationError):
        LLMExplanation(
            short_title="x" * 101,
            rationale="Test",
            fields_referenced=["field1"],
            risk_level="low",
            suggested_action="Test",
            anomaly_type="other"
        )
    
    # Rationale too long (>800 chars)
    with pytest.raises(ValidationError):
        LLMExplanation(
            short_title="Test",
            rationale="x" * 801,
            fields_referenced=["field1"],
            risk_level="low",
            suggested_action="Test",
            anomaly_type="other"
        )
    
    # Suggested action too long (>300 chars)
    with pytest.raises(ValidationError):
        LLMExplanation(
            short_title="Test",
            rationale="Test",
            fields_referenced=["field1"],
            risk_level="low",
            suggested_action="x" * 301,
            anomaly_type="other"
        )


def test_llm_explanation_empty_provenance():
    """Test that provenance defaults to empty dict."""
    explanation = LLMExplanation(
        short_title="Test",
        rationale="Test",
        fields_referenced=["field1"],
        risk_level="low",
        suggested_action="Test",
        anomaly_type="other"
    )
    
    assert explanation.provenance == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

