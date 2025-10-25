"""
Page 4: Individual Explain & Review Anomalies
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from app.ui_theme import apply_custom_theme, render_page_header, render_empty_state, render_button
from app.components import render_explanation_panel, render_user_feedback_form
from core.llm_explainer import explain_anomaly
from core.shap_utils import compute_shap_values, get_top_shap_features
from core.rules import get_triggered_rules
from core.persistence import log_user_feedback
from core.schema import UserFeedback
from core.config import config

apply_custom_theme()

render_page_header(
    "Individual Explain & Review Anomalies",
    "Generate explanations for specific anomalies and provide detailed feedback",
    ""
)

# Initialize explanations in session state if not exists
if 'explanations' not in st.session_state:
    st.session_state.explanations = {}

# Ensure explanations is a proper dict
if not isinstance(st.session_state.explanations, dict):
    st.session_state.explanations = {}


# Check if detection has been run
if not st.session_state.get('detection_run', False):
    render_empty_state("Detection not run yet. Please complete detection first.", "")
    st.stop()

# Get data from session
df_raw = st.session_state.df_raw
X = st.session_state.X
meta = st.session_state.meta
scores = st.session_state.scores
flags = st.session_state.flags
rules_df = st.session_state.rules_df
detector = st.session_state.detector

# Compute SHAP values if not already done
if 'shap_values' not in st.session_state:
    with st.spinner("Computing SHAP values..."):
        shap_values = compute_shap_values(detector, X)
        st.session_state.shap_values = shap_values
else:
    shap_values = st.session_state.shap_values

# Get flagged indices
flagged_indices = flags[flags == True].index.tolist()

if len(flagged_indices) == 0:
    render_empty_state("No anomalies detected. Try lowering the threshold sensitivity.", "")
    st.stop()

st.success(f"{len(flagged_indices)} Anomalies Detected!")

# Individual anomaly review
st.markdown("### Review Individual Anomalies")

# Anomaly selector
if len(st.session_state.explanations) > 0:
    explained_indices = list(st.session_state.explanations.keys())
    default_idx = explained_indices[0]
else:
    explained_indices = []
    default_idx = flagged_indices[0]

# Use LLM checkbox for individual explanations
use_llm_individual = st.checkbox(
    "Use LLM for natural language explanation",
    value=config.has_groq_key(),
    help="Enable LLM-based explanation for this specific row (requires API key)"
)

# Show dropdown of flagged anomalies
anomaly_options = {}
for idx in flagged_indices[:50]:  # Limit to 50 for performance
    score = scores.loc[idx]
    voucher = meta.loc[idx, 'voucher_id'] if 'voucher_id' in meta.columns else idx
    
    # Add indicator if explanation already exists
    explanation_status = "✅" if idx in st.session_state.explanations else "⏳"
    anomaly_options[f"{explanation_status} Row {idx} | Score: {score:.3f} | Voucher: {voucher}"] = idx

# Initialize selected index in session state if not exists
if 'selected_anomaly_idx' not in st.session_state:
    st.session_state.selected_anomaly_idx = flagged_indices[0] if flagged_indices else None

# Find the index of the currently selected anomaly in the options
current_selected_idx = st.session_state.selected_anomaly_idx
default_index = 0
if current_selected_idx and current_selected_idx in [anomaly_options[key] for key in anomaly_options.keys()]:
    # Find the index of the current selection in the options list
    for i, key in enumerate(anomaly_options.keys()):
        if anomaly_options[key] == current_selected_idx:
            default_index = i
            break

selected_label = st.selectbox(
    "Select anomaly to review",
    options=list(anomaly_options.keys()),
    index=default_index,
    key="anomaly_selector"
)
selected_idx = anomaly_options[selected_label]

# Update session state with the selected index
st.session_state.selected_anomaly_idx = selected_idx

# Check if explanation exists for this specific index
# Only consider it exists if it's actually in the explanations dict and has valid content
explanation_exists = (selected_idx in st.session_state.explanations and 
                    st.session_state.explanations[selected_idx] is not None and
                    hasattr(st.session_state.explanations[selected_idx], 'short_title'))

if not explanation_exists:
    if render_button("Explain This Row"):
        # Generate explanation for this specific row
        row = df_raw.loc[selected_idx]
        score = scores.loc[selected_idx]
        triggered = get_triggered_rules(rules_df, selected_idx)
        
        shap_feats = []
        if shap_values is not None:
            shap_feats = get_top_shap_features(shap_values, X, selected_idx)
        
        with st.spinner("Generating explanation..."):
            explanation = explain_anomaly(
                row,
                score,
                triggered,
                shap_feats,
                df_raw.columns.tolist(),
                use_llm=use_llm_individual
            )
            st.session_state.explanations[selected_idx] = explanation
            st.session_state.individual_explanation_generated = True
            st.rerun()
else:
    # Explanation already exists - show it instantly
    # Determine if it's from batch or individual
    is_from_batch = (st.session_state.get('batch_explanations_generated', False) and 
                    selected_idx in st.session_state.get('batch_explained_indices', []))
    
    # Show regenerate button only for individual explanations
    if not is_from_batch:
        if render_button("Regenerate Explanation", type="primary"):
            # Remove existing explanation and regenerate
            del st.session_state.explanations[selected_idx]
            st.session_state.individual_explanation_generated = False
            st.rerun()
    
    # Automatically show the explanation
    st.session_state.individual_explanation_generated = True

# Show individual explanation if generated
if st.session_state.get('individual_explanation_generated', False) and selected_idx in st.session_state.explanations:
    st.divider()
    st.markdown("### Individual Generated Explanation")
    
    # Display selected row details
    st.markdown("#### Selected Anomaly Details")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Transaction Details**")
        row_data = df_raw.loc[selected_idx]
        
        # Display as formatted data
        display_data = {}
        for col in ['date', 'voucher_id', 'account', 'amount', 'debit', 'credit', 'vendor', 'poster', 'description']:
            if col in row_data.index:
                display_data[col] = row_data[col]
        
        st.json(display_data)

    with col2:
        st.markdown("**Detection Metrics**")
        st.metric("Anomaly Score", f"{scores.loc[selected_idx]:.3f}")
        st.metric("Threshold", f"{st.session_state.threshold:.3f}")
        
        # Triggered rules
        triggered = get_triggered_rules(rules_df, selected_idx)
        if triggered:
            st.markdown("**Triggered Rules:**")
            for rule in triggered:
                st.markdown(f"- {rule}")
        else:
            st.markdown("*No rules triggered*")

    st.divider()
    
    # Show explanation
    st.markdown("#### Explanation")
    explanation = st.session_state.explanations[selected_idx]
    
    # Get SHAP features
    shap_feats = []
    if shap_values is not None:
        shap_feats = get_top_shap_features(shap_values, X, selected_idx)
    
    render_explanation_panel(explanation, shap_feats)
    
    st.divider()
    st.markdown("### User Study & Feedback - Individual Explanation")
    
    # User feedback for individual explanation
    if f'individual_feedback_submitted_{selected_idx}' not in st.session_state:
        st.markdown("**Please review the explanation above and provide your feedback:**")
        
        feedback_data = render_user_feedback_form(
            row_idx=selected_idx,
            anomaly_flag=True,
            score=scores.loc[selected_idx],
            llm_title=explanation.short_title
        )
        
        if feedback_data:
            # Add timing
            start_time = st.session_state.get(f'review_start_{selected_idx}', time.time() * 1000)
            end_time = time.time() * 1000
            ms_to_decide = int(end_time - start_time)
            
            # Create feedback object
            feedback = UserFeedback(
                row_id=feedback_data['row_id'],
                anomaly_flag=feedback_data['anomaly_flag'],
                model_score=feedback_data['model_score'],
                llm_short_title=feedback_data['llm_short_title'],
                user_is_anom=feedback_data['user_is_anom'],
                user_type=feedback_data['user_type'],
                confidence=feedback_data['confidence'],
                ms_to_decide=ms_to_decide,
                timestamp=datetime.now()
            )
            
            # Log feedback
            success = log_user_feedback(feedback)
            
            if success:
                st.success("✅ **Individual feedback submitted successfully!** Thank you for participating in the user study.")
                st.session_state[f'individual_feedback_submitted_{selected_idx}'] = True
                st.rerun()  # Rerun to hide the form and show success message
            else:
                st.error("❌ Failed to save feedback. Please try again.")
    else:
        st.success("✅ **Individual feedback submitted successfully!** Thank you for participating in the user study.")
    
    # Track review start time for the selected anomaly
    if f'review_start_{selected_idx}' not in st.session_state:
        st.session_state[f'review_start_{selected_idx}'] = time.time() * 1000

# Explanation Summary - Always at the bottom
if st.session_state.explanations:
    st.divider()
    st.markdown("### 📋 Explanation Summary")
    st.info(f"**{len(st.session_state.explanations)} explanations generated** across both batch and individual generation methods.")

if st.session_state.explanations:
    col1, col2, col3 = st.columns(3)
    
    explanations = st.session_state.explanations
    
    with col1:
        st.metric("Explanations Generated", len(explanations))
    
    with col2:
        risk_counts = {}
        for exp in explanations.values():
            risk_counts[exp.risk_level] = risk_counts.get(exp.risk_level, 0) + 1
        high_risk_count = risk_counts.get('high', 0)
        st.metric("High Risk", high_risk_count)
    
    with col3:
        type_counts = {}
        for exp in explanations.values():
            type_counts[exp.anomaly_type] = type_counts.get(exp.anomaly_type, 0) + 1
        most_common_type = max(type_counts, key=type_counts.get) if type_counts else "None"
        st.metric("Most Common Type", most_common_type)

# Sidebar info
with st.sidebar:
    st.markdown("### Individual Analysis")
    st.markdown("""
    **Individual Explanation Features:**
    - Analyze specific anomalies in detail
    - SHAP feature importance per anomaly
    - Risk assessment and classification
    - Anomaly type identification
    - Personalized user feedback
    
    **Analysis Capabilities:**
    - Row-by-row examination
    - Feature contribution analysis
    - Rule-based triggers review
    - Contextual explanations
    """)
    
    st.divider()
    
    st.markdown("### Configuration")
    st.markdown("""
    **Current Settings:**
    - XGBoost: {}
    - SHAP: {}
    - LLM: {}
    - API Key: {}
    """.format(
        "Enabled" if config.ENABLE_XGBOOST else "Disabled",
        "Enabled" if config.ENABLE_SHAP else "Disabled", 
        "Enabled" if config.ENABLE_LLM else "Disabled",
        "Configured" if config.has_groq_key() else "Not configured"
    ))
