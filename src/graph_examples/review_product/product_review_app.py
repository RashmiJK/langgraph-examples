import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage

from graph_examples.logger import get_logger
from graph_examples.review_product.editorial_board import EditorialBoard

load_dotenv(override=True)
logger = get_logger(__name__)

# page config
st.set_page_config(
    page_title="AI Product Comparison",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "#### AI Product Comparison Studio\nPowered by LangGraph, OpenAI, and Deepgram.",
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
# session state: final_audio
if "final_audio" not in st.session_state:
    st.session_state.final_audio = None


image_file1_path = Path(__file__).parent / "media" / "ai_product_comp_studio.png"
image_file2_path = Path(__file__).parent / "media" / "workflow.png"

# sidebar
with st.sidebar:
    st.image(image_file1_path, width=300)
    st.markdown("---")

    st.markdown("## 🤖 Workflow Status")
    st.write("")
    st.markdown("**🎯 Chief Editor** - receives your request.")
    st.write("")
    st.markdown("**🔍 Research Team** - finds and scrapes product info")
    st.write("")
    st.markdown("**🎬 Production Team** - writes and voices the script")
    st.markdown("---")
    st.markdown("⚡️Powered by LangGraph Multi-Agent System")
    st.image(image_file2_path, width=300)


# Main content
st.markdown(
    """
<div style='text-align: center; background: white; padding: 1rem; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-bottom: 2rem;'>
    <h2 style='color: #667eea; margin: 0;'>🎙️ AI Product Comparision Studio 🔈 </h2>
    <p style='color: #666; margin-top: 0.5rem; font-size: 1rem;'>
        Enter the product names, and watch the AI research, write, and narrate a review for you
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
    <div style='align-items: center; justify-content: center; text-align: center; height: 40px; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem;'>
        <h5 style='color: #667eea; margin-top: 0;'>📝 First Product</h5>
    </div>
    """,
        unsafe_allow_html=True,
    )
    product_a = st.text_input(
        "First Product",
        placeholder="e.g., iPhone 15 Pro",
        key="product_a",
        label_visibility="collapsed",
    )

with col2:
    st.markdown(
        """
    <div style='align-items: center; justify-content: center; text-align: center; height: 40px; background: white; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem;'>
        <h5  style='color: #667eea; margin-top: 0;'>🆚 Second Product</h5>
    </div>
    """,
        unsafe_allow_html=True,
    )
    product_b = st.text_input(
        "Second Product",
        placeholder="e.g., Samsung Galaxy S24",
        key="product_b",
        label_visibility="collapsed",
    )
st.write("---")
# position compare button
col_centre = st.columns([1, 2, 1])[1]
with col_centre:
    if st.button("🚀 Compare Products", width="stretch", type="primary"):
        if product_a and product_b and product_a != product_b:
            tracer_name = Path(__file__).parent.name
            editorial_board = EditorialBoard(trace_project_name=tracer_name)

            query = f"Compare {product_a} and {product_b}. Which is better?"

            with st.status(
                "⏳🧐 AI Agents are researching & recording ...", expanded=True
            ) as status:
                # placeholder for the latest action
                latest_action = st.empty()

                PREFIX = "Audio generation complete:"

                for namespace, chunk_msg in editorial_board.stream_workflow(query):
                    if isinstance(namespace, tuple) and len(namespace):
                        names = [m.split(":")[0] for m in namespace]
                        display_msg = "➡️".join(names)
                    else:
                        display_msg = "🔄 Analysis in progress..."

                    if (
                        isinstance(chunk_msg, tuple)
                        and len(chunk_msg)
                        and isinstance(chunk_msg[0], BaseMessage)
                    ):
                        content = chunk_msg[0].content
                        if content and PREFIX in content:
                            st.session_state.final_audio = content.split(PREFIX, 1)[
                                -1
                            ].strip()

                    latest_action.markdown(f"{display_msg}")

                    logger.info(f"Processed: {display_msg}")

                status.update(
                    label="✅ Analysis completed!", state="complete", expanded=False
                )

            st.session_state.analysis_completed = True
            if st.session_state.final_audio:
                st.balloons()
            st.rerun()
        else:
            st.error("⚠️ Please enter two different products to compare")

# Display results
if st.session_state.analysis_completed:
    final_audio_path, final_script_path = None, None
    if st.session_state.final_audio:
        final_audio_path = Path(__file__).parent / st.session_state.final_audio
        final_script_path = Path(final_audio_path).with_suffix(".txt")

    if final_audio_path and final_script_path:
        st.divider()
        st.subheader("🌟 Final Production")

        cols = st.columns([1, 1.5])

        with cols[0]:
            if final_audio_path and os.path.exists(final_audio_path):
                st.success("✅ Audio Ready!")
                st.audio(final_audio_path)

                # Download button
            with open(final_audio_path, "rb") as file:
                st.download_button(
                    label="Download MP3",
                    data=file,
                    file_name=final_audio_path.name,
                    mime="audio/mp3",
                )
        with cols[1]:
            with st.expander("Read the Full Script", expanded=True, icon="📖"):
                if final_script_path and os.path.exists(final_script_path):
                    with open(final_script_path, "r") as f:
                        script_content = f.read()
                    st.markdown(
                        f"""
        <div style='height: 300px; overflow-y: scroll; border: 1px solid #ddd; padding: 10px; border-radius: 5px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);'>
            {script_content}
        </div>
        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("Script content unavailable.")

            # Download script
            if final_script_path and os.path.exists(final_script_path):
                with open(final_script_path, "r") as file:
                    st.download_button(
                        label="Download Script",
                        data=file,
                        file_name=final_script_path.name,
                        mime="text/plain",
                    )
    else:
        st.error("""
    ⚠️ **Something went wrong during production.**
    This is likely due to:
    1. **GitHub Models Rate Limit**: The free tier has strict request limits.
    2. **Context Overflow**: Too much research data gathered.
    **Debugging:**
    *   Check your terminal or Opik traces for `RateLimitError` or `ContextWindowExceeded`.
    *   Try a simpler query or wait a moment before retrying.
    """)

    if st.button("🔄 New Comparison", use_container_width=True, type="primary"):
        # Reset session state
        st.session_state.analysis_completed = False
        st.session_state.product_a = None
        st.session_state.product_b = None
        st.session_state.final_audio = None
        st.rerun()

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    """
<div style='text-align: center; color: rgba(255,255,255,0.7); padding: 2rem;'>
    <p> 🌟🤖 | 🎙️ Production Studio powered by LangGraph Agents, Deepgram & Streamlit 🎬  | 🤖✨ </p>
</div>
""",
    unsafe_allow_html=True,
)
