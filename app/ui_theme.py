"""
UI theme and styling configuration for Streamlit.
"""
import streamlit as st


def apply_custom_theme():
    """Apply custom CSS and theme configuration."""
    
    st.set_page_config(
        page_title="Anomaly Detection & XAI",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        /* Main container */
        .main > div {
            padding-top: 2rem;
        }
        
        /* KPI Cards */
        .kpi-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .kpi-card-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        .kpi-card-label {
            font-size: 0.9rem;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        /* Risk badges */
        .risk-high {
            background-color: #ef4444;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
        }
        
        .risk-medium {
            background-color: #f59e0b;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
        }
        
        .risk-low {
            background-color: #10b981;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
        }
        
        /* Info boxes */
        .info-box {
            background-color: #f0f9ff;
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        
        .warning-box {
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        
        .success-box {
            background-color: #f0fdf4;
            border-left: 4px solid #10b981;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        
        /* Headers */
        h1 {
            color: #1e293b;
            font-weight: 700;
        }
        
        h2 {
            color: #334155;
            font-weight: 600;
            margin-top: 2rem;
        }
        
        h3 {
            color: #475569;
            font-weight: 600;
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.2s;
            min-height: 40px;
            padding: 0.5rem 1.5rem;
            font-size: 0.95rem;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        /* Primary buttons */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            border: none;
            color: white;
        }
        
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
            box-shadow: 0 6px 16px rgba(59, 130, 246, 0.3);
        }
        
        /* Secondary buttons */
        .stButton > button[kind="secondary"] {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border: 1px solid #cbd5e1;
            color: #475569;
        }
        
        .stButton > button[kind="secondary"]:hover {
            background: linear-gradient(135deg, #f1f5f9 0%, #d1d5db 100%);
            border-color: #94a3b8;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        /* Success buttons */
        .stButton > button[style*="background-color: rgb(34, 197, 94)"] {
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%) !important;
            border: none !important;
            color: white !important;
        }
        
        .stButton > button[style*="background-color: rgb(34, 197, 94)"]:hover {
            background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
            box-shadow: 0 6px 16px rgba(34, 197, 94, 0.3) !important;
        }
        
        /* Download buttons */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
            border: none;
            color: white;
            min-height: 40px;
            padding: 0.5rem 1.5rem;
            font-size: 0.95rem;
            border-radius: 8px;
            font-weight: 500;
        }
        
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
            box-shadow: 0 6px 16px rgba(139, 92, 246, 0.3);
            transform: translateY(-2px);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #f8fafc;
        }
        
        /* Metric cards in Streamlit */
        [data-testid="stMetricValue"] {
            font-size: 2rem;
        }
        
        /* Table styling */
        .dataframe {
            font-size: 0.9rem;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #334155;
        }
    </style>
    """, unsafe_allow_html=True)


def render_page_header(title: str, description: str, icon: str = ""):
    """Render a consistent page header."""
    st.markdown(f"# {icon} {title}")
    st.markdown(f"<p style='font-size: 1.1rem; color: #64748b; margin-bottom: 2rem;'>{description}</p>", 
                unsafe_allow_html=True)
    st.divider()


def render_empty_state(message: str, icon: str = ""):
    """Render an empty state message."""
    st.markdown(f"""
    <div style='text-align: center; padding: 4rem 2rem;'>
        <h3 style='color: #64748b; font-weight: 500;'>{message}</h3>
    </div>
    """, unsafe_allow_html=True)


def render_loading_state(message: str = "Processing..."):
    """Render a loading state."""
    with st.spinner(message):
        st.empty()


def render_button(text: str, type: str = "primary", use_container_width: bool = True, key: str = None, **kwargs):
    """Render a consistently styled button."""
    return st.button(
        text, 
        type=type, 
        use_container_width=use_container_width, 
        key=key,
        **kwargs
    )


def render_download_button(label: str, data, file_name: str, mime: str = "text/csv", key: str = None):
    """Render a consistently styled download button."""
    return st.download_button(
        label=label,
        data=data,
        file_name=file_name,
        mime=mime,
        key=key,
        use_container_width=True
    )

