import time

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

# page config
st.set_page_config(
    page_title="Product Review Analyzer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "#### Analyze product reviews from multiple sources and generate comprehensive reports",
    },
)

# inject custome CSS into Streamlit page
st.markdown(
    """
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Card-like containers */
    .stApp {
        background: transparent;
    }
    
    /* Custom metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: white;
    }
    
    div[data-testid="metric-container"] label {
        color: rgba(255, 255, 255, 0.9) !important;
        font-weight: 600;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2rem !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Input fields */
    .stTextInput input {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 12px;
        font-size: 16px;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        font-weight: 600;
        color: #667eea;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""",
    unsafe_allow_html=True,
)

# session state : analysis_completed
if "analysis_completed" not in st.session_state:
    st.session_state.analysis_completed = False
if "product_data" not in st.session_state:
    st.session_state.product_data = None

# sidebar
with st.sidebar:
    st.markdown("# 🔍 Product Review Analyzer")
    st.markdown("---")

    st.markdown("---")
    st.markdown("### 📊 Focus Areas")
    st.markdown("Sentiment Analysis")
    st.markdown("Feature Extraction")
    st.markdown("Price Tracking")

    st.markdown("---")
    st.markdown("### 🎨 Display Options")
    st.selectbox("Chart Theme", ["Plotly", "Streamlit", "Dark"], key="theme")
    st.checkbox("Generate Word Cloud", value=True, key="show_word_cloud")

# Main content
st.markdown(
    """
<div style='background: white; padding: 2rem; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-bottom: 2rem;'>
    <h1 style='color: #667eea; margin: 0;'>🔍 Product Review Analyzer</h1>
    <p style='color: #666; margin-top: 0.5rem; font-size: 1.1rem;'>
        Compare products with AI-powered sentiment analysis and insights
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Layout, two columns
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
    <div style='background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
        <h3 style='color: #667eea; margin-top: 0;'>📱 Product A</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )
    product_a = st.text_input(
        "",
        placeholder="e.g., iPhone 15 Pro",
        key="product_a",
        label_visibility="collapsed",
    )

with col2:
    st.markdown(
        """
    <div style='background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem;'>
        <h3 style='color: #667eea; margin-top: 0;'>📱 Product B</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )
    product_b = st.text_input(
        "",
        placeholder="e.g., Samsung Galaxy S24",
        key="product_b",
        label_visibility="collapsed",
    )

# position analyze button
col_centre = st.columns([1, 2, 1])[1]
with col_centre:
    if st.button("🚀 Analyze Products", width="stretch"):
        if product_a and product_b and product_a != product_b:
            show_progress_container = st.container()
            with show_progress_container:
                st.markdown(
                    """
                <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 2rem 0;'>
                    <h3 style='color: #667eea; text-align: center;'>⚙️ Analyzing Products...</h3>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                progress_bar = st.progress(0)
                progress_status = st.empty()

                steps = [
                    "🔍 Searching for products...",
                    "🌐 Scraping reviews from Amazon...",
                    "💬 Collecting Reddit discussions...",
                    "🎯 Analyzing sentiment...",
                    "✨ Extracting key features...",
                    "📊 Generating insights...",
                ]

                for i, step in enumerate(steps):
                    progress_status.markdown(
                        f"<p style='text-align: center; color: #667eea; font-weight: 600;'>{step}</p>",
                        unsafe_allow_html=True,
                    )
                    progress_bar.progress((i + 1) / len(steps))
                    time.sleep(0.5)

                st.session_state.analysis_completed = True
                st.session_state.product_data = {
                    "product_a": product_a,
                    "product_b": product_b,
                }
                # show_progress_container.empty() # .empty() don't seem to empty the container
                st.rerun()  # Right now button press is assisting in hiding the progress container
        else:
            st.error("⚠️ Please enter two different products to compare")

# Display results
if st.session_state.analysis_completed:
    st.markdown("---")

    # Key Metrics
    st.markdown(
        """
    <div style='background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem;'>
        <h2 style='color: #667eea; margin-top: 0;'>📊 Key Insights</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label=f"{st.session_state.product_data['product_a']} Rating",
            value="4.6 ⭐",
            delta="0.3",
        )

    with col2:
        st.metric(
            label=f"{st.session_state.product_data['product_b']} Rating",
            value="4.4 ⭐",
            delta="-0.1",
        )

    with col3:
        st.metric(label="Total Reviews Analyzed", value="847", delta="234 new")

    with col4:
        st.metric(label="Confidence Score", value="92%", delta="High")

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Sentiment Analysis",
            "🎯 Feature Comparison",
            "💰 Price Insights",
            "📝 Top Reviews",
        ]
    )


# Session State Debugging
for key, value in st.session_state.items():
    st.write(f"{key}: {value}")
st.write(f"Product A: {product_a} \\t Product B: {product_b}")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
<div style='text-align: center; color: rgba(255,255,255,0.7); padding: 2rem;'>
    <p> 🌟🤖 | Backend using LangGraph Multi-Agent System | Frontend using Streamlit | 🤖✨ </p>
</div>
""",
    unsafe_allow_html=True,
)
