"""
Pydantic schemas for data validation and LLM outputs.
"""
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class LLMExplanation(BaseModel):
    """Schema for LLM-generated anomaly explanations."""
    
    short_title: str = Field(
        ...,
        description="Brief, actionable title (< 100 chars)",
        max_length=100
    )
    
    rationale: str = Field(
        ...,
        description="Detailed explanation of why this is flagged",
        max_length=800
    )
    
    fields_referenced: List[str] = Field(
        ...,
        description="List of field names that were key to the decision"
    )
    
    risk_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Severity assessment"
    )
    
    suggested_action: str = Field(
        ...,
        description="Recommended next steps for auditor",
        max_length=300
    )
    
    anomaly_type: Literal[
        "duplicate",
        "misclassification",
        "unbalanced",
        "timing_policy",
        "other"
    ] = Field(
        ...,
        description="Category of the anomaly"
    )
    
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supporting evidence: rules triggered, SHAP values, scores"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "short_title": "Duplicate transaction on same day",
                "rationale": "This entry has an identical amount and vendor as another transaction posted on the same date, which is unusual for this account.",
                "fields_referenced": ["amount", "vendor", "date", "account"],
                "risk_level": "medium",
                "suggested_action": "Verify with vendor that only one transaction should exist; check for processing error.",
                "anomaly_type": "duplicate",
                "provenance": {
                    "rules_triggered": ["duplicate_amount_same_day"],
                    "model_score": 0.87,
                    "shap_top_features": ["vendor_freq", "amount_z", "duplicate_flag"]
                }
            }
        }
    )


class UserFeedback(BaseModel):
    """Schema for user study feedback."""
    
    row_id: int = Field(..., description="Index of the row being reviewed")
    anomaly_flag: bool = Field(..., description="Whether model flagged as anomaly")
    model_score: float = Field(..., description="Model's anomaly score")
    llm_short_title: Optional[str] = Field(None, description="LLM explanation title")
    
    user_is_anom: bool = Field(..., description="User's judgment: is this an anomaly?")
    user_type: Optional[str] = Field(None, description="If anomaly, what type?")
    confidence: int = Field(..., ge=1, le=7, description="User confidence (1-7)")
    ms_to_decide: int = Field(..., description="Milliseconds taken to decide")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "row_id": 42,
                "anomaly_flag": True,
                "model_score": 0.89,
                "llm_short_title": "Weekend transaction in unusual account",
                "user_is_anom": True,
                "user_type": "timing_policy",
                "confidence": 6,
                "ms_to_decide": 8500,
                "timestamp": "2025-10-23T10:30:00"
            }
        }
    )


class LedgerRow(BaseModel):
    """Schema for a ledger entry."""
    
    date: str
    voucher_id: str
    account: str
    debit: Optional[float] = None
    credit: Optional[float] = None
    amount: float
    vendor: Optional[str] = None
    poster: Optional[str] = None
    description: Optional[str] = None
    
    model_config = ConfigDict(extra="allow")  # Allow additional fields

