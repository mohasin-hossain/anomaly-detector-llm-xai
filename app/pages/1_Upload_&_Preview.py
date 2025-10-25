"""
Page 1: Data Upload and Preview
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
from app.ui_theme import apply_custom_theme, render_page_header, render_empty_state, render_button
from core.features import build_features
from core.config import config

apply_custom_theme()

# Clear the features_just_built flag on page load
if 'features_just_built' in st.session_state:
    del st.session_state.features_just_built

render_page_header(
    "Upload & Preview Data",
    "Load your accounting ledger data and preview the feature engineering pipeline",
    ""
)

# File uploader
st.markdown("### Data Source")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload a CSV file with columns: date, voucher_id, account, debit, credit, amount, vendor, poster, description"
)

use_sample = st.checkbox("Use sample data", value=True)

# Load data
df_raw = None

if use_sample:
    try:
        df_raw = pd.read_csv('data/sample_ledger.csv')
        st.success(f"Loaded sample data: {len(df_raw)} rows")
    except FileNotFoundError:
        st.error("Sample data not found. Please upload a CSV file.")
        use_sample = False

if uploaded_file is not None and not use_sample:
    try:
        df_raw = pd.read_csv(uploaded_file)
        st.success(f"Uploaded file loaded: {len(df_raw)} rows")
    except Exception as e:
        st.error(f"Failed to load file: {e}")

# Display data if loaded
if df_raw is not None:
    st.divider()
    
    # Data audit
    st.markdown("### Data Audit")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df_raw):,}")
    with col2:
        st.metric("Columns", len(df_raw.columns))
    with col3:
        missing_pct = (df_raw.isnull().sum().sum() / (len(df_raw) * len(df_raw.columns)) * 100)
        st.metric("Missing Values", f"{missing_pct:.1f}%")
    with col4:
        if 'amount' in df_raw.columns:
            total_amount = df_raw['amount'].abs().sum()
            st.metric("Total Amount", f"${total_amount:,.0f}")
    
    # Preview raw data
    st.markdown("### Raw Data Preview")
    preview_rows = min(50, len(df_raw))  # Show up to 50 rows or all if less
    st.markdown(f"*Showing first {preview_rows} rows of {len(df_raw)} total rows*")
    st.dataframe(df_raw.head(preview_rows), use_container_width=True, height=400)
    
    # Column info
    with st.expander("Column Information"):
        col_info = pd.DataFrame({
            'Column': df_raw.columns,
            'Type': df_raw.dtypes.values,
            'Non-Null': df_raw.count().values,
            'Null Count': df_raw.isnull().sum().values,
            'Unique': [df_raw[col].nunique() for col in df_raw.columns]
        })
        st.dataframe(col_info, use_container_width=True)
    
    st.divider()
    
    # Feature engineering
    
    if render_button("Build Features", type="primary"):
        with st.spinner("Building features..."):
            try:
                X, feature_cols, meta, flags_df = build_features(df_raw)
                
                # Store in session state
                st.session_state.df_raw = df_raw
                st.session_state.X = X
                st.session_state.feature_cols = feature_cols
                st.session_state.meta = meta
                st.session_state.flags_df = flags_df
                st.session_state.data_loaded = True
                st.session_state.total_rows = len(df_raw)
                st.session_state.features_just_built = True
                
                st.success(f"Features built successfully! {len(feature_cols)} features created.")
                
                # Show feature summary
                st.markdown("#### Feature Statistics")
                
                st.dataframe(X.describe().T[['mean', 'std', 'min', 'max']].head(10), 
                           use_container_width=True)
                
                # Feature preview
                st.markdown("#### Feature Matrix Preview")
                display_features = X.head(10).copy()
                display_features.index = meta.head(10)['voucher_id'].values if 'voucher_id' in meta.columns else display_features.index
                st.dataframe(display_features, use_container_width=True)
                
                st.info("Features are ready! Proceed to the **Detect Anomalies** page to run detection.")
                
            except Exception as e:
                st.error(f"Feature engineering failed: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Show cached features if available (but not immediately after building)
    if st.session_state.get('data_loaded', False) and not st.session_state.get('features_just_built', False):
        st.success("Features already built and cached in session!")
        
        with st.expander("View Cached Features"):
            st.markdown(f"**Feature count:** {len(st.session_state.feature_cols)}")
            st.markdown(f"**Row count:** {len(st.session_state.X)}")
            st.dataframe(st.session_state.X.head(), use_container_width=True)

else:
    render_empty_state("No data loaded yet. Upload a CSV or use sample data to get started.", "")

# Sidebar info
with st.sidebar:
    st.markdown("### Data Requirements")
    st.markdown("""
    **Expected CSV Columns:**
    - `date`: Transaction date
    - `voucher_id`: Voucher/transaction ID
    - `account`: Account code
    - `debit`: Debit amount
    - `credit`: Credit amount
    - `amount`: Transaction amount
    - `vendor`: Vendor name
    - `poster`: User who posted
    - `description`: Transaction description
    
    *Note: Some columns are optional*
    """)
    
    st.divider()
    
    st.markdown("### Tips")
    st.markdown("""
    - Ensure dates are in a standard format
    - Amounts should be numeric
    - Missing values are handled automatically
    - Sample data includes 8 injected anomalies
    """)

