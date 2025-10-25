"""
Page 4: User Study Results and Analytics
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.ui_theme import apply_custom_theme, render_page_header, render_empty_state, render_button, render_download_button
from app.components import render_confusion_matrix, render_time_distribution
from core.persistence import load_user_study_results, compute_user_study_metrics, export_results_summary
from core.config import config

apply_custom_theme()

render_page_header(
    "User Study Results",
    "Analyze model performance and user feedback metrics",
    ""
)

# Load results
results_df = load_user_study_results()

if results_df is None or len(results_df) == 0:
    render_empty_state(
        "No user feedback collected yet. Review some anomalies in the 'Explain & Review' page first.",
        ""
    )
    st.stop()


# Compute metrics
metrics = compute_user_study_metrics(results_df)

# KPI Cards
st.markdown("### Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Reviews",
        metrics['total_reviews'],
        help="Total number of user feedback entries"
    )

with col2:
    precision = metrics.get('precision', None)
    if precision is not None:
        st.metric(
            "Precision",
            f"{precision:.1%}",
            help="Of flagged items, how many were true anomalies?"
        )
    else:
        st.metric("Precision", "N/A")

with col3:
    recall = metrics.get('recall', None)
    if recall is not None:
        st.metric(
            "Recall",
            f"{recall:.1%}",
            help="Of true anomalies, how many were flagged?"
        )
    else:
        st.metric("Recall", "N/A")

with col4:
    f1 = metrics.get('f1_score', None)
    if f1 is not None:
        st.metric(
            "F1 Score",
            f"{f1:.3f}",
            help="Harmonic mean of precision and recall"
        )
    else:
        st.metric("F1 Score", "N/A")

st.divider()

# Agreement and confidence
st.markdown("### Agreement & Confidence")

col1, col2 = st.columns(2)

with col1:
    agreement_rate = metrics.get('agreement_rate', 0)
    st.metric(
        "Model-User Agreement",
        f"{agreement_rate:.1%}",
        help="Percentage of times model and user agreed"
    )
    
    # Agreement visualization
    agreement_data = pd.DataFrame({
        'Category': ['Agree', 'Disagree'],
        'Count': [
            int(agreement_rate * metrics['total_reviews']),
            int((1 - agreement_rate) * metrics['total_reviews'])
        ]
    })
    
    fig = px.pie(
        agreement_data,
        names='Category',
        values='Count',
        title="Model-User Agreement",
        color='Category',
        color_discrete_map={'Agree': '#10b981', 'Disagree': '#ef4444'}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    avg_confidence = metrics.get('avg_confidence', 0)
    st.metric(
        "Average Confidence",
        f"{avg_confidence:.2f} / 7",
        help="Average user confidence rating (1-7 scale)"
    )
    
    # Confidence distribution
    confidence_counts = results_df['confidence'].value_counts().sort_index()
    
    fig = go.Figure(data=[
        go.Bar(
            x=confidence_counts.index,
            y=confidence_counts.values,
            marker_color='#667eea'
        )
    ])
    
    fig.update_layout(
        title="Confidence Distribution",
        xaxis_title="Confidence Level (1-7)",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Confusion matrix
st.markdown("### Confusion Matrix")

render_confusion_matrix(results_df)

st.divider()

# Decision time analysis
st.markdown("### Decision Time Analysis")

col1, col2 = st.columns(2)

with col1:
    avg_time = metrics.get('avg_ms_to_decide', 0)
    st.metric(
        "Average Time",
        f"{avg_time/1000:.2f}s",
        help="Average time to make a decision"
    )

with col2:
    median_time = metrics.get('median_ms_to_decide', 0)
    st.metric(
        "Median Time",
        f"{median_time/1000:.2f}s",
        help="Median time to make a decision"
    )

render_time_distribution(results_df)

st.divider()

# Anomaly type distribution
st.markdown("### Anomaly Type Distribution")

type_dist = metrics.get('anomaly_type_distribution', {})

if type_dist:
    type_df = pd.DataFrame({
        'Anomaly Type': list(type_dist.keys()),
        'Count': list(type_dist.values())
    })
    
    fig = px.bar(
        type_df,
        x='Anomaly Type',
        y='Count',
        title="User-Labeled Anomaly Types",
        color='Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No anomaly type data available yet.")

st.divider()

# Detailed results table
st.markdown("### Detailed Results")

with st.expander("View All Feedback Entries"):
    display_df = results_df.copy()
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['seconds_to_decide'] = (display_df['ms_to_decide'] / 1000).round(2)
    st.dataframe(display_df, use_container_width=True, height=400)

st.divider()

# Score analysis
st.markdown("### Score Analysis")

col1, col2 = st.columns(2)

with col1:
    # True positives vs false positives - score comparison
    tp_df = results_df[(results_df['anomaly_flag'] == True) & (results_df['user_is_anom'] == True)]
    fp_df = results_df[(results_df['anomaly_flag'] == True) & (results_df['user_is_anom'] == False)]
    
    fig = go.Figure()
    
    if len(tp_df) > 0:
        fig.add_trace(go.Box(
            y=tp_df['model_score'],
            name='True Positives',
            marker_color='#10b981'
        ))
    
    if len(fp_df) > 0:
        fig.add_trace(go.Box(
            y=fp_df['model_score'],
            name='False Positives',
            marker_color='#ef4444'
        ))
    
    fig.update_layout(
        title="Model Scores: TP vs FP",
        yaxis_title="Anomaly Score",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Confidence by correctness
    correct = results_df[results_df['anomaly_flag'] == results_df['user_is_anom']]
    incorrect = results_df[results_df['anomaly_flag'] != results_df['user_is_anom']]
    
    fig = go.Figure()
    
    if len(correct) > 0:
        fig.add_trace(go.Box(
            y=correct['confidence'],
            name='Model Correct',
            marker_color='#10b981'
        ))
    
    if len(incorrect) > 0:
        fig.add_trace(go.Box(
            y=incorrect['confidence'],
            name='Model Incorrect',
            marker_color='#ef4444'
        ))
    
    fig.update_layout(
        title="User Confidence by Model Accuracy",
        yaxis_title="Confidence (1-7)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Export
st.markdown("### Export Results")

col1, col2 = st.columns(2)

with col1:
    # Export enhanced CSV
    csv_data = results_df.to_csv(index=False)
    render_download_button(
        label="Download Results (CSV)",
        data=csv_data,
        file_name="user_study_results.csv"
    )

with col2:
    # Export summary metrics
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': [str(v) for v in metrics.values()]
    })
    
    metrics_csv = metrics_df.to_csv(index=False)
    render_download_button(
        label="Download Metrics Summary (CSV)",
        data=metrics_csv,
        file_name="metrics_summary.csv"
    )

st.divider()

# Insights and recommendations
st.markdown("### Insights & Recommendations")

insights = []

# Precision analysis
if metrics.get('precision', None) is not None:
    precision = metrics['precision']
    if precision >= 0.8:
        insights.append(("Excellent Precision", f"The model achieves {precision:.1%} precision. Very few false positives!"))
    elif precision >= 0.6:
        insights.append(("Moderate Precision", f"Precision is {precision:.1%}. Consider increasing threshold to reduce false positives."))
    else:
        insights.append(("Low Precision", f"Precision is only {precision:.1%}. The threshold may be too low, causing many false alarms."))

# Recall analysis
if metrics.get('recall', None) is not None:
    recall = metrics['recall']
    if recall >= 0.8:
        insights.append(("Excellent Recall", f"The model catches {recall:.1%} of true anomalies. Good coverage!"))
    elif recall >= 0.6:
        insights.append(("Moderate Recall", f"Recall is {recall:.1%}. Some anomalies might be missed."))
    else:
        insights.append(("Low Recall", f"Recall is only {recall:.1%}. Consider lowering threshold to catch more anomalies."))

# Confidence analysis
if avg_confidence < 4.0:
    insights.append(("Low Confidence", "Users have low confidence in their assessments. Consider improving explanations."))
elif avg_confidence >= 5.5:
    insights.append(("High Confidence", "Users are confident in their assessments. Explanations are effective!"))

# Display insights
for title, message in insights:
    if "Excellent" in title or "High" in title:
        st.success(f"**{title}**\n\n{message}")
    elif "Moderate" in title or "Low" in title:
        st.warning(f"**{title}**\n\n{message}")

# Sidebar
with st.sidebar:
    st.markdown("### Metrics Explained")
    st.markdown("""
    **Precision**
    - TP / (TP + FP)
    - Of flagged items, how many are real anomalies?
    
    **Recall**
    - TP / (TP + FN)
    - Of real anomalies, how many did we catch?
    
    **F1 Score**
    - 2 × (P × R) / (P + R)
    - Balanced metric
    
    **Agreement Rate**
    - How often model and user agree
    """)
    

