"""
Page 3: Batch Explain & Review Anomalies
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
from core.llm_explainer import batch_explain
from core.shap_utils import compute_shap_values, get_top_shap_features
from core.rules import get_triggered_rules
from core.persistence import log_user_feedback
from core.schema import UserFeedback
from core.config import config

apply_custom_theme()

render_page_header(
    "Batch Explain & Review Anomalies",
    "Generate explanations for multiple anomalies at once and provide batch feedback",
    ""
)

# Initialize explanations in session state if not exists
if 'explanations' not in st.session_state:
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

st.success(f"{len(flagged_indices)} Anomalies Detected! ")

# Batch explanation
st.markdown("### Batch Explanation")

use_llm = st.checkbox(
    "Use LLM for natural language explanation",
    value=config.has_groq_key(),
    help="Enable LLM-based explanations (requires API key)"
)

top_n = st.number_input(
    "Explain top N anomalies",
    min_value=1,
    max_value=min(50, len(flagged_indices)),
    value=min(10, len(flagged_indices)),
    help="Number of top-scored anomalies to explain"
)

if render_button("Generate Explanations", type="primary"):
    # Get top N flagged rows by score
    top_indices = scores[flags].nlargest(top_n).index.tolist()
    
    # Build SHAP features map
    shap_features_map = {}
    if shap_values is not None:
        for idx in top_indices:
            shap_features_map[idx] = get_top_shap_features(shap_values, X, idx)
    
    with st.spinner(f"Generating explanations for {len(top_indices)} anomalies..."):
        explanations = batch_explain(
            df_raw,
            scores,
            rules_df,
            shap_features_map,
            top_indices,
            use_llm=use_llm
        )
        
        # Store in session
        st.session_state.explanations.update(explanations)
        st.session_state.batch_explanations_generated = True
        st.session_state.batch_explained_indices = top_indices  # Track which indices were explained in batch
        st.success(f"Generated {len(explanations)} explanations!")
        st.rerun()

# Show batch explanations if generated
if st.session_state.get('batch_explanations_generated', False) and st.session_state.explanations:
    st.divider()
    st.markdown("### Batch Generated Explanations")
    
    # Show all generated explanations
    explained_indices = list(st.session_state.explanations.keys())
    st.info(f"📋 **{len(explained_indices)} explanations generated** for the top {len(explained_indices)} anomalies.")
    
    # Display all explanations
    for i, idx in enumerate(explained_indices):
        explanation = st.session_state.explanations[idx]
        
        st.markdown(f"#### Explanation {i+1} of {len(explained_indices)} - Row {idx}")
        
        # Get SHAP features for this explanation
        shap_feats = []
        if shap_values is not None:
            shap_feats = get_top_shap_features(shap_values, X, idx)
        
        render_explanation_panel(explanation, shap_feats)
        
        # Add separator between explanations (except for the last one)
        if i < len(explained_indices) - 1:
            st.divider()
    
    st.divider()
    st.markdown("### User Study & Feedback - Batch Explanations")
    
    # Let user select which explanation to provide feedback for
    feedback_options = {}
    for idx in explained_indices:
        score = scores.loc[idx]
        voucher = meta.loc[idx, 'voucher_id'] if 'voucher_id' in meta.columns else idx
        feedback_options[f"Row {idx} | Score: {score:.3f} | Voucher: {voucher}"] = idx
    
    selected_feedback_idx = st.selectbox(
        "Select explanation to provide feedback for:",
        options=list(feedback_options.keys()),
        index=0,
        key="batch_feedback_selector"
    )
    selected_feedback_idx = feedback_options[selected_feedback_idx]
    
    # User feedback for selected explanation
    if f'batch_feedback_submitted_{selected_feedback_idx}' not in st.session_state:
        st.markdown(f"**Please review the explanation for Row {selected_feedback_idx} and provide your feedback:**")
        
        explanation = st.session_state.explanations[selected_feedback_idx]
        feedback_data = render_user_feedback_form(
            row_idx=selected_feedback_idx,
            anomaly_flag=True,
            score=scores.loc[selected_feedback_idx],
            llm_title=explanation.short_title
        )
        
        if feedback_data:
            # Add timing
            start_time = st.session_state.get(f'review_start_{selected_feedback_idx}', time.time() * 1000)
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
                st.success("✅ **Batch feedback submitted successfully!** Thank you for participating in the user study.")
                st.session_state[f'batch_feedback_submitted_{selected_feedback_idx}'] = True
                st.rerun()  # Rerun to hide the form and show success message
            else:
                st.error("❌ Failed to save feedback. Please try again.")
    else:
        st.success("✅ **Batch feedback submitted successfully!** Thank you for participating in the user study.")
    
    # Track review start time for the selected feedback
    if f'review_start_{selected_feedback_idx}' not in st.session_state:
        st.session_state[f'review_start_{selected_feedback_idx}'] = time.time() * 1000

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
    
    st.markdown("### Batch Processing")
    st.markdown("""
    **Batch Explanation Features:**
    - Generate explanations for multiple anomalies
    - SHAP feature importance analysis
    - Risk level classification
    - Anomaly type categorization
    - User feedback collection
    
    **LLM Integration:**
    - Groq Llama-3.3-70b-versatile
    - Natural language explanations
    - JSON-structured output
    - Template fallbacks available
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
