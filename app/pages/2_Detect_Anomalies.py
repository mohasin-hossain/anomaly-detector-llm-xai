"""
Page 2: Anomaly Detection
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
from app.ui_theme import apply_custom_theme, render_page_header, render_empty_state, render_button, render_download_button
from app.components import render_kpi_cards, render_score_distribution
from core.detectors import AnomalyDetector
from core.rules import apply_rules
from core.config import config

apply_custom_theme()

render_page_header(
    "Detect Anomalies",
    "Run the detection pipeline and analyze flagged transactions",
    ""
)

# Check if data is loaded
if not st.session_state.get('data_loaded', False):
    render_empty_state("No data loaded. Please go to Upload & Preview first.", "")
    st.stop()

# Get data from session
df_raw = st.session_state.df_raw
X = st.session_state.X
feature_cols = st.session_state.feature_cols
meta = st.session_state.meta

st.markdown("### Detection Configuration")

# Threshold sensitivity slider
col1, col2 = st.columns([3, 1])

with col1:
    threshold_k = st.slider(
        "Threshold Sensitivity (k)",
        min_value=1.0,
        max_value=5.0,
        value=float(config.THRESHOLD_K),
        step=0.5,
        help="Multiplier for MAD in threshold calculation. Lower = more sensitive."
    )

with col2:
    st.empty()  # Empty space to balance the layout


# Checkboxes on new line
col1, col2, col3 = st.columns(3)

with col1:
    enable_xgb = st.checkbox(
        "Enable XGBoost",
        value=config.ENABLE_XGBOOST,
        help="Use XGBoost for typicality scoring"
    )

with col2:
    enable_rules = st.checkbox(
        "Enable Rules",
        value=True,
        help="Include rule-based detection"
    )

with col3:
    st.empty()  # Empty space to balance the layout

# Run detection
if render_button("Run Detection", type="primary"):
    with st.spinner("Running anomaly detection..."):
        try:
            # Apply rules
            rules_df, rule_scores = apply_rules(df_raw, X)
            
            # Initialize detector
            detector = AnomalyDetector()
            
            # Fit models
            st.info("Training IsolationForest...")
            detector.fit(X, feature_cols)
            
            # Predict
            st.info("Computing anomaly scores...")
            scores, flags, threshold = detector.predict(
                X, 
                rule_scores if enable_rules else np.zeros(len(X)),
                rules_df,
                k=threshold_k
            )
            
            # Store results in session
            st.session_state.scores = pd.Series(scores, index=X.index)
            st.session_state.flags = pd.Series(flags, index=X.index)
            st.session_state.threshold = threshold
            st.session_state.rules_df = rules_df
            st.session_state.rule_scores = rule_scores
            st.session_state.detector = detector
            st.session_state.detection_run = True
            st.session_state.flagged_count = flags.sum()
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Detection failed: {e}")
            import traceback
            st.code(traceback.format_exc())

# Display results if detection has been run
if st.session_state.get('detection_run', False):
    st.divider()
    
    scores = st.session_state.scores
    flags = st.session_state.flags
    threshold = st.session_state.threshold
    rules_df = st.session_state.rules_df
    
    # KPI cards
    st.markdown("### Detection Summary")
    render_kpi_cards(
        flagged_count=int(flags.sum()),
        total_count=len(flags),
        threshold=threshold,
        precision=None  # Will be computed from user feedback later
    )
    
    st.divider()
    
    # Score distribution
    st.markdown("### Score Distribution")
    render_score_distribution(scores, threshold, flags)
    
    st.divider()
    
    # Anomaly table
    st.markdown("### Flagged Anomalies")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"Showing top anomalies sorted by score (max {config.MAX_ROWS_DISPLAY} rows)")
    
    with col2:
        show_all = st.checkbox("Show all records", value=False)
    
    # Build display DataFrame
    display_df = meta.copy()
    display_df['anomaly_score'] = scores
    display_df['is_anomaly'] = flags
    
    if show_all:
        # Show all records
        display_df = display_df.sort_values('anomaly_score', ascending=False).head(config.MAX_ROWS_DISPLAY)
    else:
        # Show flagged only
        display_df = display_df[display_df['is_anomaly'] == True].sort_values('anomaly_score', ascending=False).head(config.MAX_ROWS_DISPLAY)
    
    if len(display_df) == 0:
        st.info("No anomalies detected with current threshold.")
    else:
        # Color legend
        st.markdown("**Color Legend:**")
        st.markdown("""
        <div style="display: flex; gap: 10px; margin-bottom: 10px; flex-wrap: wrap;">
            <span style="background-color: #fee2e2; padding: 4px 8px; border-radius: 4px; font-size: 12px; border: 1px solid #fecaca;">🔴 High Risk (≥0.8)</span>
            <span style="background-color: #fef3c7; padding: 4px 8px; border-radius: 4px; font-size: 12px; border: 1px solid #fde68a;">🟡 Medium Risk (0.5-0.8)</span>
            <span style="background-color: #f8f9fa; padding: 4px 8px; border-radius: 4px; font-size: 12px; border: 1px solid #dee2e6;">⚪ Low Risk (<0.5)</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Color-code scores
        def highlight_scores(row):
            if row['anomaly_score'] >= 0.8:
                return ['background-color: #fee2e2'] * len(row)
            elif row['anomaly_score'] >= 0.5:
                return ['background-color: #fef3c7'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = display_df.style.apply(highlight_scores, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=500)
    
    st.divider()
    
    # Rule summary
    st.markdown("### Rule Trigger Summary")
    
    rule_summary = pd.DataFrame({
        'Rule': rules_df.columns,
        'Triggered Count': rules_df.sum().values,
        'Percentage': (rules_df.sum() / len(rules_df) * 100).round(2).values
    }).sort_values('Triggered Count', ascending=False)
    
    st.dataframe(rule_summary, use_container_width=True, height=300)
    
    st.divider()
    
    # Download results
    st.markdown("### Export Results")
    
    export_df = meta.copy()
    export_df['anomaly_score'] = scores
    export_df['is_anomaly'] = flags
    export_df = export_df[export_df['is_anomaly'] == True].sort_values('anomaly_score', ascending=False)
    
    csv = export_df.to_csv(index=False)
    render_download_button(
        label="Download Flagged Anomalies (CSV)",
        data=csv,
        file_name="flagged_anomalies.csv"
    )
    
    st.info("Detection complete! Proceed to **Explain & Review** to get detailed explanations.")

else:
    st.info("Click 'Run Detection' to start analyzing your data.")

# Sidebar info
with st.sidebar:
    st.markdown("### Detection Methods")
    st.markdown("""
    **1. IsolationForest**
    - Unsupervised anomaly detection
    - Isolates outliers in feature space
    
    **2. XGBoost (optional)**
    - Predicts typical amounts
    - Flags high residuals
    
    **3. Rule-Based**
    - Business logic checks
    - Unbalanced vouchers
    - Duplicates
    - Timing violations
    
    **4. Ensemble**
    - Weighted combination
    - Robust thresholding (MAD)
    """)
    
    st.divider()
    
    st.markdown("### Tips")
    st.markdown("""
    - Lower k = more sensitive
    - Higher k = fewer false positives
    - Try k=2.0 for aggressive detection
    - Try k=3.0 for conservative detection
    """)

