"""
Reusable UI components for Streamlit app.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List, Dict
from core.schema import LLMExplanation
from app.ui_theme import render_button


def render_kpi_cards(flagged_count: int, total_count: int, threshold: float, precision: Optional[float] = None):
    """Render KPI cards in columns."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Records",
            value=f"{total_count:,}",
            help="Total number of ledger entries"
        )
    
    with col2:
        st.metric(
            label="Flagged Anomalies",
            value=f"{flagged_count:,}",
            delta=f"{(flagged_count/total_count*100):.1f}%" if total_count > 0 else "0%",
            help="Number of entries flagged as anomalies"
        )
    
    with col3:
        st.metric(
            label="Detection Threshold",
            value=f"{threshold:.3f}",
            help="Threshold computed using Median + k*MAD"
        )
    
    with col4:
        if precision is not None:
            st.metric(
                label="Precision (est.)",
                value=f"{precision:.1%}",
                help="Estimated precision from user feedback"
            )
        else:
            st.metric(
                label="Precision",
                value="N/A",
                help="No user feedback yet"
            )


def render_score_distribution(scores: pd.Series, threshold: float, flags: pd.Series):
    """Render histogram of anomaly scores."""
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=50,
        name="All Scores",
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {threshold:.3f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Distribution of Anomaly Scores",
        xaxis_title="Anomaly Score",
        yaxis_title="Count",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)



def render_explanation_panel(
    explanation: LLMExplanation,
    shap_features: List[Dict] = None
):
    """Render explanation in a side panel or expander."""
    
    # Title with risk badge
    risk_class = f"risk-{explanation.risk_level}"
    st.markdown(f"""
    <h3>{explanation.short_title}</h3>
    <span class="{risk_class}">{explanation.risk_level.upper()} RISK</span>
    <span style="margin-left: 1rem; color: #64748b; font-size: 0.9rem;">
        Type: {explanation.anomaly_type}
    </span>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Rationale
    st.markdown("**Why This Was Flagged:**")
    st.write(explanation.rationale)
    
    # Fields referenced
    st.markdown("**Fields Referenced:**")
    st.write(", ".join(f"`{f}`" for f in explanation.fields_referenced))
    
    # Suggested action
    st.markdown("**Suggested Action:**")
    st.info(explanation.suggested_action)
    
    # Provenance
    with st.expander("Technical Details"):
        st.json(explanation.provenance)
    
    # SHAP features if available
    if shap_features:
        with st.expander("Feature Contributions (SHAP)"):
            shap_df = pd.DataFrame(shap_features)
            st.dataframe(shap_df, use_container_width=True)
            
            # Bar chart
            fig = go.Figure(go.Bar(
                x=shap_df['shap_contribution'],
                y=shap_df['feature'],
                orientation='h',
                marker_color=['red' if x < 0 else 'green' for x in shap_df['shap_contribution']]
            ))
            fig.update_layout(
                title="SHAP Feature Contributions",
                xaxis_title="SHAP Value",
                yaxis_title="Feature",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)


def render_user_feedback_form(row_idx: int, anomaly_flag: bool, score: float, 
                               llm_title: Optional[str] = None) -> Optional[Dict]:
    """
    Render user feedback form.
    
    Returns:
        Dict with feedback data or None
    """
    st.markdown("### Your Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        user_is_anom = st.radio(
            "Is this an anomaly?",
            options=[True, False],
            format_func=lambda x: "Yes, it's an anomaly" if x else "No, it's normal",
            key=f"feedback_anom_{row_idx}"
        )
    
    with col2:
        confidence = st.slider(
            "Confidence (1-7)",
            min_value=1,
            max_value=7,
            value=4,
            help="How confident are you in this assessment?",
            key=f"feedback_conf_{row_idx}"
        )
    
    user_type = None
    if user_is_anom:
        user_type = st.selectbox(
            "Anomaly type",
            options=["duplicate", "misclassification", "unbalanced", "timing_policy", "other"],
            key=f"feedback_type_{row_idx}"
        )
    
    submitted = render_button("Submit Feedback", type="primary", key=f"feedback_submit_{row_idx}")
    
    if submitted:
        return {
            'row_id': row_idx,
            'anomaly_flag': anomaly_flag,
            'model_score': score,
            'llm_short_title': llm_title,
            'user_is_anom': user_is_anom,
            'user_type': user_type,
            'confidence': confidence
        }
    
    return None


def render_confusion_matrix(results_df: pd.DataFrame):
    """Render confusion matrix from user study results."""
    if results_df is None or len(results_df) == 0:
        return
    
    # Compute confusion matrix
    tp = ((results_df['anomaly_flag'] == True) & (results_df['user_is_anom'] == True)).sum()
    fp = ((results_df['anomaly_flag'] == True) & (results_df['user_is_anom'] == False)).sum()
    tn = ((results_df['anomaly_flag'] == False) & (results_df['user_is_anom'] == False)).sum()
    fn = ((results_df['anomaly_flag'] == False) & (results_df['user_is_anom'] == True)).sum()
    
    # Create heatmap
    cm = [[tn, fp], [fn, tp]]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Normal', 'Predicted Anomaly'],
        y=['Actual Normal', 'Actual Anomaly'],
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        colorscale='Blues',
        showscale=False
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Model Prediction",
        yaxis_title="User Label",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_time_distribution(results_df: pd.DataFrame):
    """Render distribution of decision times."""
    if results_df is None or len(results_df) == 0:
        return
    
    fig = px.histogram(
        results_df,
        x='ms_to_decide',
        nbins=30,
        title="Distribution of Decision Times",
        labels={'ms_to_decide': 'Time to Decide (ms)'}
    )
    
    # Add median line
    median_time = results_df['ms_to_decide'].median()
    fig.add_vline(
        x=median_time,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Median: {median_time:.0f}ms"
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_risk_pie_chart(explanations: Dict[int, LLMExplanation]):
    """Render pie chart of risk levels."""
    if not explanations:
        return
    
    risk_counts = {}
    for exp in explanations.values():
        risk_counts[exp.risk_level] = risk_counts.get(exp.risk_level, 0) + 1
    
    fig = go.Figure(data=[go.Pie(
        labels=list(risk_counts.keys()),
        values=list(risk_counts.values()),
        marker_colors=['#10b981', '#f59e0b', '#ef4444']
    )])
    
    fig.update_layout(
        title="Risk Level Distribution",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

