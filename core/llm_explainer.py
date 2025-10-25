"""
LLM-based natural language explanation using Groq.
"""
import json
import os
from typing import Dict, Any, List, Optional
import pandas as pd
from pydantic import ValidationError

from core.config import config
from core.schema import LLMExplanation

# Conditional Groq import
if config.has_groq_key():
    try:
        from groq import Groq
        groq_client = Groq(api_key=config.GROQ_API_KEY)
    except ImportError:
        groq_client = None
else:
    groq_client = None


def build_system_prompt() -> str:
    """Build system prompt with JSON schema enforcement."""
    schema_example = {
        "short_title": "Brief title describing the anomaly",
        "rationale": "Detailed explanation citing specific field values",
        "fields_referenced": ["field1", "field2"],
        "risk_level": "low | medium | high",
        "suggested_action": "What the auditor should do next",
        "anomaly_type": "duplicate | misclassification | unbalanced | timing_policy | other",
        "provenance": {
            "rules_triggered": [],
            "shap_top_features": [],
            "model_score": 0.0
        }
    }
    
    return f"""You are an accounting anomaly explanation engine.

Your task: Analyze flagged ledger entries and produce clear, actionable explanations.

CRITICAL RULES:
1. Output ONLY valid JSON. No markdown, no code blocks, no extra text.
2. Use this exact schema:
{json.dumps(schema_example, indent=2)}

3. Be concise but specific - cite actual field values from the row.
4. Choose anomaly_type from: duplicate, misclassification, unbalanced, timing_policy, other
5. Set risk_level based on severity: low, medium, high
6. In suggested_action, be practical and actionable.
7. In rationale, explain WHY this is suspicious using the evidence provided.

Remember: JSON only, no markdown formatting."""


def build_user_prompt(
    row: pd.Series,
    score: float,
    rules_triggered: List[str],
    shap_features: List[Dict],
    schema_fields: List[str]
) -> str:
    """Build user prompt with row context and evidence."""
    
    # Format row data (limit to key fields)
    row_data = {
        k: (str(v) if pd.notna(v) else "N/A")
        for k, v in row.to_dict().items()
        if k in schema_fields[:10]  # Limit to avoid token bloat
    }
    
    # Format SHAP features
    shap_summary = []
    for feat in shap_features[:3]:
        shap_summary.append(
            f"{feat['feature']}={feat['value']:.2f} (SHAP: {feat['shap_contribution']:+.3f})"
        )
    
    prompt = f"""CONTEXT
- Schema fields available: {', '.join(schema_fields[:15])}
- Model anomaly score: {score:.3f}
- Triggered rules: {', '.join(rules_triggered) if rules_triggered else 'None'}
- Top SHAP features: {', '.join(shap_summary) if shap_summary else 'N/A'}

ROW DATA
{json.dumps(row_data, indent=2)}

TASK
Analyze this flagged entry and output a JSON explanation following the schema.
Focus on the most suspicious aspects. Be specific about field values."""
    
    return prompt


def call_groq_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = None,
    max_tokens: int = None
) -> Optional[str]:
    """
    Call Groq API with prompts.
    
    Returns:
        Raw JSON string or None on failure
    """
    if groq_client is None:
        return None
    
    if temperature is None:
        temperature = config.GROQ_TEMPERATURE
    if max_tokens is None:
        max_tokens = config.GROQ_MAX_TOKENS
    
    try:
        response = groq_client.chat.completions.create(
            model=config.GROQ_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        raw_content = response.choices[0].message.content
        return raw_content.strip()
    
    except Exception as e:
        print(f"Groq API call failed: {e}")
        return None


def parse_and_validate(raw_json: str) -> Optional[LLMExplanation]:
    """
    Parse JSON and validate with Pydantic.
    
    Args:
        raw_json: Raw JSON string from LLM
        
    Returns:
        Validated LLMExplanation or None
    """
    try:
        # Try to extract JSON if wrapped in markdown
        if "```json" in raw_json:
            raw_json = raw_json.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_json:
            raw_json = raw_json.split("```")[1].split("```")[0].strip()
        
        data = json.loads(raw_json)
        explanation = LLMExplanation(**data)
        return explanation
    
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Validation failed: {e}")
        return None


def create_fallback_explanation(
    row: pd.Series,
    score: float,
    rules_triggered: List[str],
    shap_features: List[Dict]
) -> LLMExplanation:
    """
    Create template-based explanation when LLM unavailable.
    """
    # Determine anomaly type from rules
    anomaly_type = "other"
    if "duplicate_same_day" in rules_triggered:
        anomaly_type = "duplicate"
    elif "unbalanced_voucher" in rules_triggered:
        anomaly_type = "unbalanced"
    elif "outside_business_hours" in rules_triggered:
        anomaly_type = "timing_policy"
    
    # Determine risk level from score
    if score >= 0.8:
        risk_level = "high"
    elif score >= 0.5:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    # Build title
    if rules_triggered:
        short_title = f"Flagged: {', '.join(rules_triggered[:2])}"
    else:
        short_title = f"Statistical anomaly (score: {score:.2f})"
    
    # Build rationale
    rationale_parts = []
    if rules_triggered:
        rationale_parts.append(f"Business rules triggered: {', '.join(rules_triggered)}.")
    if shap_features:
        top_feat = shap_features[0]
        rationale_parts.append(
            f"Key factor: {top_feat['feature']} = {top_feat['value']:.2f}."
        )
    rationale_parts.append(f"Overall anomaly score: {score:.2f}.")
    
    rationale = " ".join(rationale_parts)
    
    # Fields referenced
    fields_ref = ["amount", "date"]
    if shap_features:
        fields_ref.extend([f['feature'] for f in shap_features[:2]])
    
    # Suggested action
    suggested_action = (
        "Review transaction details and verify with supporting documentation. "
        "Contact relevant parties if discrepancies found."
    )
    
    return LLMExplanation(
        short_title=short_title[:100],
        rationale=rationale[:800],
        fields_referenced=list(set(fields_ref))[:5],
        risk_level=risk_level,
        suggested_action=suggested_action,
        anomaly_type=anomaly_type,
        provenance={
            "rules_triggered": rules_triggered,
            "shap_top_features": [f['feature'] for f in shap_features[:3]],
            "model_score": score,
            "source": "template_fallback"
        }
    )


def explain_anomaly(
    row: pd.Series,
    score: float,
    rules_triggered: List[str],
    shap_features: List[Dict],
    schema_fields: List[str],
    use_llm: bool = True
) -> LLMExplanation:
    """
    Main function to generate explanation for an anomaly.
    
    Args:
        row: The data row
        score: Anomaly score
        rules_triggered: List of triggered rule names
        shap_features: Top SHAP features
        schema_fields: Available field names
        use_llm: Whether to use LLM (False = template only)
        
    Returns:
        LLMExplanation object
    """
    # If LLM disabled or no API key, use fallback
    if not use_llm or not config.has_groq_key() or groq_client is None:
        return create_fallback_explanation(row, score, rules_triggered, shap_features)
    
    # Build prompts
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(row, score, rules_triggered, shap_features, schema_fields)
    
    # Call LLM
    raw_json = call_groq_llm(system_prompt, user_prompt)
    
    if raw_json is None:
        return create_fallback_explanation(row, score, rules_triggered, shap_features)
    
    # Parse and validate
    explanation = parse_and_validate(raw_json)
    
    if explanation is None:
        return create_fallback_explanation(row, score, rules_triggered, shap_features)
    
    # Enrich provenance
    explanation.provenance.update({
        "rules_triggered": rules_triggered,
        "shap_top_features": [f['feature'] for f in shap_features[:3]],
        "model_score": score,
        "source": "groq_llm"
    })
    
    return explanation


def batch_explain(
    df: pd.DataFrame,
    scores: pd.Series,
    rules_df: pd.DataFrame,
    shap_features_map: Dict[int, List[Dict]],
    indices: List[int],
    use_llm: bool = True
) -> Dict[int, LLMExplanation]:
    """
    Generate explanations for multiple rows.
    
    Args:
        df: Full DataFrame
        scores: Anomaly scores
        rules_df: Rules DataFrame
        shap_features_map: Map of index -> SHAP features
        indices: List of row indices to explain
        use_llm: Whether to use LLM
        
    Returns:
        Dict mapping index to explanation
    """
    explanations = {}
    schema_fields = df.columns.tolist()
    
    for idx in indices:
        if idx not in df.index:
            continue
        
        row = df.loc[idx]
        score = scores.loc[idx]
        rules_triggered = [col for col in rules_df.columns if rules_df.loc[idx, col]]
        shap_features = shap_features_map.get(idx, [])
        
        explanation = explain_anomaly(
            row, score, rules_triggered, shap_features, schema_fields, use_llm
        )
        explanations[idx] = explanation
    
    return explanations

