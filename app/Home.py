"""
Main Streamlit application entry point.
Multi-page app for anomaly detection and explanation.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from app.ui_theme import apply_custom_theme, render_page_header
from core.config import config

# Apply theme
apply_custom_theme()

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'detection_run' not in st.session_state:
    st.session_state.detection_run = False
if 'explanations' not in st.session_state:
    st.session_state.explanations = {}
if 'total_rows' not in st.session_state:
    st.session_state.total_rows = 0
if 'flagged_count' not in st.session_state:
    st.session_state.flagged_count = 0
if 'batch_explained_indices' not in st.session_state:
    st.session_state.batch_explained_indices = []
if 'selected_anomaly_idx' not in st.session_state:
    st.session_state.selected_anomaly_idx = None

# Check if this is the first visit to the application (only redirect if user hasn't been to Upload page yet)
if 'app_initialized' not in st.session_state and 'data_loaded' not in st.session_state:
    st.session_state.app_initialized = True
    # Redirect to Upload & Preview page on first visit to the app
    st.switch_page("pages/1_Upload_&_Preview.py")

# Main page (only shown if user navigates back to Home)
render_page_header(
    "Anomaly Detection & Explanation",
    "Sophisticated anomaly detection system combines cutting-edge machine learning with explainable AI to identify suspicious patterns in accounting ledger data ",
    ""
)

st.markdown("""
### Core Capabilities

- **Multi-Model Detection**: Combines IsolationForest, XGBoost, and rule-based systems for comprehensive coverage
- **Explainable AI**: SHAP value analysis and LLM-generated explanations provide transparency into detection decisions
- **Business Rule Engine**: Implements accounting-specific rules for common anomaly patterns
- **Interactive Analysis**: Real-time threshold adjustment and detailed investigation tools
- **User Study Framework**: Captures expert feedback for continuous model improvement

### Workflow

1. **Upload & Preview**: Import your accounting ledger CSV or explore with sample data
2. **Detect Anomalies**: Execute multi-model detection with configurable sensitivity
3. **Batch Explain & Review**: Generate explanations for multiple anomalies at once
4. **Individual Explain & Review**: Analyze specific anomalies with detailed explanations
5. **User Study Results**: Review detection performance and user feedback insights

### System Configuration

""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Detection Settings:**
    - IsolationForest contamination: {:.1%}
    - Threshold method: Median + k×MAD
    - Ensemble weights: ISO {:.0%} | XGB {:.0%} | Rules {:.0%}
    """.format(
        config.IFOREST_CONTAMINATION,
        config.WEIGHT_IFOREST,
        config.WEIGHT_XGBOOST,
        config.WEIGHT_RULES
    ))

with col2:
    st.markdown("""
    **Feature Flags:**
    - XGBoost: {}
    - SHAP: {}
    - LLM Explanations: {}
    - API Key configured: {}
    """.format(
        "Yes" if config.ENABLE_XGBOOST else "No",
        "Yes" if config.ENABLE_SHAP else "No",
        "Yes" if config.ENABLE_LLM else "No",
        "Yes" if config.has_groq_key() else "No"
    ))

if not config.has_groq_key():
    st.warning("""
    **Groq API Key Not Configured**
    
    LLM explanations will use template-based fallbacks. To enable full LLM explanations:
    1. Create a `.env` file in the project root
    2. Add your Groq API key: `GROQ_API_KEY=your_key_here`
    3. Restart the application
    
    Get a free API key at: https://console.groq.com
    """)

st.divider()

st.markdown("""
### Technical Specifications

- **AI Model**: Groq Llama-3.3-70b-versatile for natural language explanations
- **Detection Models**: IsolationForest, XGBoost, and custom rule-based systems
- **Privacy & Security**: Local data processing with optional cloud-based explanation generation
- **Performance**: Optimized for large-scale accounting datasets with real-time analysis

### Getting Started

Use the navigation menu in the sidebar to begin your analysis. Start with **Upload & Preview** to load your accounting data, then proceed through the detection and explanation workflow.
""")

# Sidebar info
with st.sidebar:
    st.markdown("### Session Status")
    
    # Data loaded status
    if st.session_state.data_loaded:
        st.markdown('<div style="margin: 8px 0; display: flex; align-items: center;"><span style="color: #22c55e; margin-right: 8px; font-size: 12px;">●</span><span>Data loaded ({})</span></div>'.format(st.session_state.get('total_rows', 0)), unsafe_allow_html=True)
    else:
        st.markdown('<div style="margin: 8px 0; display: flex; align-items: center;"><span style="color: #ef4444; margin-right: 8px; font-size: 12px;">●</span><span>No data loaded yet</span></div>', unsafe_allow_html=True)
    
    # Detection status
    if st.session_state.detection_run:
        st.markdown('<div style="margin: 8px 0; display: flex; align-items: center;"><span style="color: #22c55e; margin-right: 8px; font-size: 12px;">●</span><span>Detection complete ({})</span></div>'.format(st.session_state.get('flagged_count', 0)), unsafe_allow_html=True)
    else:
        st.markdown('<div style="margin: 8px 0; display: flex; align-items: center;"><span style="color: #ef4444; margin-right: 8px; font-size: 12px;">●</span><span>Detection not run yet</span></div>', unsafe_allow_html=True)
    
    # Explanations status
    if st.session_state.explanations:
        st.markdown('<div style="margin: 8px 0; display: flex; align-items: center;"><span style="color: #22c55e; margin-right: 8px; font-size: 12px;">●</span><span>Explanations generated ({})</span></div>'.format(len(st.session_state.explanations)), unsafe_allow_html=True)
    else:
        st.markdown('<div style="margin: 8px 0; display: flex; align-items: center;"><span style="color: #ef4444; margin-right: 8px; font-size: 12px;">●</span><span>No explanations yet</span></div>', unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### About")
    st.markdown("""
    **Version**: 1.0.0
    
    Built with:
    - Streamlit
    - scikit-learn
    - XGBoost
    - SHAP
    - Groq API
    """)

