import re
import sys
from pathlib import Path

import streamlit as st
import torch
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.crop_doctor.inference import load_model, predict_image
from src.crop_doctor.transforms import create_eval_transform
from src.crop_doctor.llm import LLMConfigError, chat_with_expert, generate_diagnosis_explanation
from src.crop_doctor.prompts import LANGUAGE_OPTIONS


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Crop Doctor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS / CONSTANTS
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "cnn"
    / "crop_doctor_baseline.pth"
)

USER_INITIALS = "MN"          # swap for the logged-in user's initials
HIGH_CONF_THRESHOLD = 0.80
MED_CONF_THRESHOLD = 0.50

LANGUAGE_FLAGS = {
    "English": "🌐",
    "Hindi": "🇮🇳",
    "German": "🇩🇪",
    "Spanish": "🇪🇸",
}


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault("language", LANGUAGE_OPTIONS[0])
st.session_state.setdefault("diagnosis", None)          # {"crop": str, "disease": str, "confidence": float, "display": str}
st.session_state.setdefault("chat_messages", [])         # [{"role": ..., "content": ...}]  — for display only
st.session_state.setdefault("chat_interaction_id", None)  # server-side conversation memory (Interactions API)
st.session_state.setdefault("chat_diagnosis_key", None)  # tracks which diagnosis the chat belongs to
st.session_state.setdefault("llm_cache", {})


def cached_llm_text(cache_key, compute_fn):
    """Avoid re-calling the LLM on every Streamlit rerun for the same inputs."""
    cache = st.session_state["llm_cache"]
    if cache_key not in cache:
        cache[cache_key] = compute_fn()
    return cache[cache_key]


def render_review_meta(review_meta: dict):
    """Small transparency panel showing what the generator->critic->reviser agent did."""
    if review_meta["revised"]:
        st.markdown(
            "<div class='explain-caption'>🔁 Self-reviewed — the first draft failed a check, "
            "so the agent revised it.</div>",
            unsafe_allow_html=True,
        )
    elif review_meta["initial_passed"]:
        st.markdown(
            "<div class='explain-caption'>✅ Self-reviewed — passed the critic's checks on "
            "the first draft.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='explain-caption'>⚠️ Self-reviewed — flagged an issue but reached "
            "the revision limit.</div>",
            unsafe_allow_html=True,
        )

    # Left as native st.markdown here on purpose: st.expander is a Streamlit
    # component with its own theme-matched background (dark bg in dark mode),
    # unlike our hard-coded-white .panel div — forcing dark text here could
    # make it invisible against a dark expander body instead.
    with st.expander("🔍 Show self-review details"):
        st.markdown(f"- **Passed on first draft:** {review_meta['initial_passed']}")
        st.markdown(f"- **Was revised:** {review_meta['revised']}")
        st.markdown(f"- **Critic's final feedback:** {review_meta['feedback']}")


def render_llm_error(exc: Exception):
    """
    Show a short, readable message for an LLM call failure, with the raw
    exception tucked behind an expander instead of dumped inline. Detects
    Gemini's 429 / quota-exceeded response specifically, since that's a
    rate limit (not a bug) and usually tells you how long to wait.
    """
    text = str(exc)

    if isinstance(exc, LLMConfigError):
        st.warning(f"⚙️ {text}")
        return

    if "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower():
        retry_match = re.search(r"retry in ([\d.]+)\s*s", text, re.IGNORECASE)
        wait_hint = f" — retry in about {float(retry_match.group(1)):.0f}s" if retry_match else ""
        st.warning(
            f"⏳ Hit Gemini's rate limit{wait_hint}. This is a free-tier quota "
            "(a handful of requests per minute), not a bug in the app — wait a "
            "moment and it'll work again."
        )
    else:
        st.warning("Couldn't generate this right now — see details below.")

    with st.expander("Show error details"):
        st.code(text)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    /* Tell the browser this page is explicitly light-themed. Without this,
       some browsers/OS settings auto-invert or force-dark unlabeled pages —
       which happens at the rendering layer, AFTER our CSS is computed, so
       no amount of explicit color/!important on our end can fix it. This is
       almost certainly the real cause of the washed-out text. */
    html, body, .stApp {
        color-scheme: light only;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background-color: #f7faf8;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 4.2rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background-color: #f7faf8;
        border-bottom: 1px solid #dce8e0;
        z-index: 999;
    }

    #MainMenu, footer {visibility: hidden;}


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eafaf0 0%, #dcf0e4 55%, #cfe9da 100%);
        border-right: 1px solid #d8e8de;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.4rem;
    }

    .sidebar-brand-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 2px;
    }

    .sidebar-brand-icon {
        font-size: 26px;
    }

    .sidebar-brand-title {
        font-size: 22px;
        font-weight: 800;
        color: #143b29;
    }

    .sidebar-brand-subtitle {
        font-size: 12px;
        color: #23804f;
        margin: 0 0 22px 36px;
    }

    section[data-testid="stSidebar"] .stButton button {
        width: 100%;
        min-height: 44px;
        border-radius: 10px;
        border: 1px solid transparent;
        background-color: transparent;
        color: #35473d;
        font-size: 14px;
        font-weight: 600;
        text-align: left;
        padding-left: 14px;
        box-shadow: none;
    }

    section[data-testid="stSidebar"] .stButton button:hover:enabled {
        background-color: rgba(255,255,255,0.55);
        border-color: #cfe4d7;
        color: #168149;
    }

    section[data-testid="stSidebar"] .stButton button:disabled {
        color: #9aa89f;
        opacity: 0.65;
    }

    div[data-testid="stSidebarNav"] {display: none;}

    .nav-active button {
        background-color: #ffffff !important;
        border: 1px solid #cfe4d7 !important;
        color: #168149 !important;
        box-shadow: 0 1px 3px rgba(20, 59, 41, 0.08);
    }

    .sidebar-illustration {
        margin-top: 40px;
        border-radius: 16px;
        height: 130px;
        position: relative;
        overflow: hidden;
        background: linear-gradient(180deg, #cdeadb 0%, #a9d9bf 100%);
    }

    .sidebar-illustration .hill-back {
        position: absolute;
        bottom: -18px;
        left: -10px;
        width: 140%;
        height: 70px;
        border-radius: 50%;
        background: #9fd2b5;
    }

    .sidebar-illustration .hill-front {
        position: absolute;
        bottom: -26px;
        left: -20px;
        width: 150%;
        height: 60px;
        border-radius: 50%;
        background: #7ec49c;
    }

    .sidebar-illustration .house {
        position: absolute;
        bottom: 30px;
        left: 18px;
        font-size: 22px;
    }

    .sidebar-illustration .tree {
        position: absolute;
        bottom: 26px;
        font-size: 18px;
    }

    .sidebar-tagline {
        margin-top: 14px;
        font-size: 12px;
        font-weight: 600;
        line-height: 1.5;
        color: #35553f;
    }


    /* ========================================================
       TOP HEADER
       ======================================================== */

    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 4px 18px 4px;
        margin-bottom: 22px;
        border-bottom: 1px solid #dce7e0;
    }

    .app-header-title {
        font-size: 18px;
        font-weight: 700;
        color: #344c3e;
    }

    .app-header-right {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .lang-pill {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 7px 14px;
        border-radius: 20px;
        border: 1px solid #dce8e0;
        background: #ffffff;
        font-size: 13px;
        color: #3c4f44;
    }

    .avatar-circle {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: #1e8a52;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero-box {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 24px;
        padding: 32px 38px;
        border-radius: 18px;
        border: 1px solid #d5eade;
        background: linear-gradient(135deg, #eef9f1 0%, #e4f5e9 100%);
        margin-bottom: 22px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        color: #14261c;
        margin-bottom: 6px;
    }

    .hero-title-green { color: #168149; }

    .hero-description {
        font-size: 15px;
        color: #53695c;
        margin-bottom: 20px;
        max-width: 520px;
    }

    .feature-pill-row {
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
    }

    .feature-pill {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 12px 16px;
        border-radius: 14px;
        min-width: 190px;
        background: rgba(255,255,255,0.6);
        border: 1px solid #d5eade;
    }

    .feature-pill.active {
        background: #ffffff;
        border-color: #bfe3cc;
        box-shadow: 0 2px 6px rgba(20,59,41,0.06);
    }

    .feature-pill .f-icon {
        font-size: 20px;
        width: 34px;
        height: 34px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #eafaf0;
    }

    .feature-pill .f-title {
        font-size: 13.5px;
        font-weight: 700;
        color: #17331f;
    }

    .feature-pill .f-desc {
        font-size: 11.5px;
        color: #66766c;
        margin-top: 2px;
    }

    .hero-right {
        min-width: 210px;
        text-align: right;
    }

    .hero-quote {
        font-size: 16px;
        font-style: italic;
        font-weight: 600;
        color: #2b4636;
        margin-bottom: 10px;
        line-height: 1.4;
    }

    .hero-plant {
        font-size: 64px;
        line-height: 1;
    }


    /* ========================================================
       CARD SHELL (used by every panel below the hero)
       ======================================================== */

    .panel {
        background: #ffffff;
        border: 1px solid #e2ece5;
        border-radius: 16px;
        padding: 20px 22px;
        height: 100%;
    }

    /* Real CSS classes for LLM-generated text — applied directly to the
       elements we render (not relying on inline style="...!important",
       which Streamlit strips from inline attributes, and not relying on a
       ".panel" ancestor selector, since that ancestor relationship isn't
       reliable across separate st.markdown() calls). No !important needed:
       a plain class beats Streamlit's own unscoped `color: inherit` rule
       through normal cascade order. */
    .explain-text {
        color: #1f2d24;
        margin-bottom: 8px;
    }

    .explain-list {
        margin: 4px 0 0 18px;
        padding: 0;
        color: #1f2d24;
    }

    .explain-list li {
        margin-bottom: 4px;
        color: #1f2d24;
    }

    .explain-caption {
        color: #5c6b61;
        font-size: 12.5px;
        margin-top: 8px;
    }

    .chat-bubble-user {
        display: flex;
        justify-content: flex-end;
        margin: 10px 0 14px 0;
    }

    .chat-bubble-user-text {
        max-width: 85%;
        background: #e8f7ed;
        color: #1f2d24;
        border-radius: 10px;
        padding: 10px 12px;
        font-size: 12.5px;
        line-height: 1.5;
    }

    .panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }

    .panel-header-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .badge-circle {
        width: 26px;
        height: 26px;
        min-width: 26px;
        border-radius: 50%;
        background: #17331f;
        color: white;
        font-size: 12px;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .panel-title {
        font-size: 16px;
        font-weight: 750;
        color: #17271e;
    }

    .panel-description {
        font-size: 12.5px;
        line-height: 1.55;
        color: #66766c;
        margin: 4px 0 14px 36px;
    }

    .conf-badge {
        font-size: 11px;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
    }

    .conf-high { background: #e4f7ea; color: #1c8a4e; }
    .conf-med  { background: #fff3d9; color: #a5750c; }
    .conf-low  { background: #fde7e7; color: #b23a3a; }


    /* ========================================================
       PREDICTION RESULT
       ======================================================== */

    .prediction-box {
        padding: 16px 18px;
        border-radius: 14px;
        border: 1px solid #ffd5d5;
        background-color: #fff5f5;
        margin: 4px 0 16px 0;
    }

    .prediction-name {
        font-size: 19px;
        font-weight: 750;
        color: #8e2525;
        margin-bottom: 2px;
    }

    .prediction-sub {
        font-size: 12px;
        color: #a86464;
        margin-bottom: 12px;
        font-style: italic;
    }

    .conf-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
    }

    .conf-label {
        font-size: 11px;
        color: #9a5c5c;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .conf-value {
        font-size: 26px;
        font-weight: 800;
        color: #d53232;
    }

    .bar-track {
        width: 100%;
        height: 9px;
        border-radius: 6px;
        background: #f6d9d9;
        overflow: hidden;
    }

    .bar-fill-red {
        height: 100%;
        border-radius: 6px;
        background: #e04b4b;
    }

    .top5-row {
        margin-bottom: 10px;
    }

    .top5-label-row {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 4px;
    }

    .top5-name { color: #2c3b31; font-weight: 600; }
    .top5-pct  { color: #66766c; font-weight: 600; }

    .bar-track-light {
        width: 100%;
        height: 7px;
        border-radius: 6px;
        background: #e9efeb;
        overflow: hidden;
    }

    .bar-fill-green {
        height: 100%;
        border-radius: 6px;
        background: #34a86a;
    }


    /* ========================================================
       DISABLED / COMING SOON PANELS
       ======================================================== */

    .panel.disabled-panel {
        background-color: #f1f4f2;
        opacity: 0.72;
    }

    .disabled-panel .panel-title,
    .disabled-panel .badge-circle {
        color: #6f7d74;
    }

    .disabled-panel .badge-circle {
        background: #b9c3bd;
    }

    .coming-soon-tag {
        display: inline-block;
        margin-top: 12px;
        font-size: 12px;
        font-weight: 700;
        color: #8b968f;
        background: #e4e9e6;
        padding: 5px 12px;
        border-radius: 20px;
    }

    .chat-bubble {
        display: flex;
        gap: 10px;
        margin: 10px 0 14px 0;
    }

    .chat-avatar {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: #d7ece0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        flex-shrink: 0;
    }

    .chat-text {
        font-size: 12.5px;
        color: #5c6b61 !important;
        background: #f5f7f6;
        border-radius: 10px;
        padding: 10px 12px;
        line-height: 1.5;
    }

    .suggested-chip {
        display: inline-block;
        font-size: 11.5px;
        color: #6f7d74;
        background: #eef1ef;
        border: 1px solid #dfe5e1;
        border-radius: 16px;
        padding: 6px 12px;
        margin: 3px 4px 0 0;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .app-footer {
        margin-top: 34px;
        padding-top: 16px;
        border-top: 1px solid #dce7e0;
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: #718078;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def format_class_name(class_name: str) -> str:
    name = class_name.replace("___", " — ")
    name = name.replace("_", " ")
    return name


def confidence_badge_html(probability: float) -> str:
    if probability >= HIGH_CONF_THRESHOLD:
        label, css_class = "High Confidence", "conf-high"
    elif probability >= MED_CONF_THRESHOLD:
        label, css_class = "Medium Confidence", "conf-med"
    else:
        label, css_class = "Low Confidence", "conf-low"
    return f'<span class="conf-badge {css_class}">{label}</span>'


def panel_header_html(number: int, title: str, right_html: str = "") -> str:
    return f"""
    <div class="panel-header">
        <div class="panel-header-left">
            <div class="badge-circle">{number}</div>
            <div class="panel-title">{title}</div>
        </div>
        {right_html}
    </div>
    """


def bullet_list_html(items) -> str:
    return "<ul class='explain-list'>" + "".join(
        f"<li>{item}</li>" for item in items
    ) + "</ul>"


def dark_text_html(text: str, bold: bool = False) -> str:
    """
    Renders text using the .explain-text CSS class (defined in the <style>
    block) instead of an inline style — Streamlit strips !important out of
    inline style="" attributes, which silently broke every earlier attempt
    at forcing text color that way. A real class isn't affected by that.
    """
    inner = f"<strong>{text}</strong>" if bold else text
    return f"<div class='explain-text'>{inner}</div>"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_crop_doctor():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, idx_to_class = load_model(MODEL_PATH, device)
    transform = create_eval_transform()
    return model, idx_to_class, transform, device


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand-row">
            <div class="sidebar-brand-icon">🌿</div>
            <div class="sidebar-brand-title">Crop Doctor</div>
        </div>
        <div class="sidebar-brand-subtitle">Healthy Crops, Brighter Tomorrow</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nav-active">', unsafe_allow_html=True)
    st.button("🏠  Home", width="stretch", key="nav_home")
    st.markdown("</div>", unsafe_allow_html=True)

    st.button("🌿  Detect Disease", width="stretch", key="nav_detect")
    st.button("💬  Ask Crop Doctor", width="stretch", disabled=True, key="nav_ask")
    st.button("📖  Crop Guide", width="stretch", disabled=True, key="nav_guide")
    st.button("ⓘ  About", width="stretch", disabled=True, key="nav_about")

    st.markdown("")

    selected_language = st.selectbox(
        "Language",
        LANGUAGE_OPTIONS,
        index=LANGUAGE_OPTIONS.index(st.session_state["language"]),
    )

    if selected_language != st.session_state["language"]:
        st.session_state["language"] = selected_language
        st.rerun()

    st.markdown(
        """
        <div class="sidebar-illustration">
            <div class="hill-back"></div>
            <div class="tree" style="left:20px;">🌳</div>
            <div class="tree" style="left:75px;">🌳</div>
            <div class="tree" style="left:130px;">🌲</div>
            <div class="house">🏡</div>
            <div class="hill-front"></div>
        </div>
        <div class="sidebar-tagline">
            🌱 Good Farmers<br/>Better Futures
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    f"""
    <div class="app-header">
        <div class="app-header-title">🌱&nbsp;&nbsp;AI for Sustainable Agriculture</div>
        <div class="app-header-right">
            <div class="lang-pill">{LANGUAGE_FLAGS.get(st.session_state["language"], "🌐")} {st.session_state["language"]}</div>
            <div class="avatar-circle">{USER_INITIALS}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO (welcome text + feature pills + quote/illustration)
# ============================================================

st.markdown(
    """
    <div class="hero-box">
        <div>
            <div class="hero-title">Welcome to <span class="hero-title-green">Crop Doctor</span></div>
            <div class="hero-description">
                Upload a plant image, get instant AI-powered diagnosis and expert guidance.
            </div>
            <div class="feature-pill-row">
                <div class="feature-pill active">
                    <div class="f-icon">🎯</div>
                    <div>
                        <div class="f-title">Detect Diseases</div>
                        <div class="f-desc">Identify plant diseases using AI.</div>
                    </div>
                </div>
                <div class="feature-pill">
                    <div class="f-icon">💡</div>
                    <div>
                        <div class="f-title">Get Guidance</div>
                        <div class="f-desc">Actionable treatment recommendations.</div>
                    </div>
                </div>
                <div class="feature-pill">
                    <div class="f-icon">🌱</div>
                    <div>
                        <div class="f-title">Learn & Prevent</div>
                        <div class="f-desc">Best practices for healthier crops.</div>
                    </div>
                </div>
                <div class="feature-pill">
                    <div class="f-icon">💬</div>
                    <div>
                        <div class="f-title">Ask Questions</div>
                        <div class="f-desc">Chat with Crop Doctor in your language.</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="hero-right">
            <div class="hero-quote">"Healthier Crops,<br/>Brighter Tomorrow"</div>
            <div class="hero-plant">🌿</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MAIN ROW: Upload | Prediction | Ask Crop Doctor (disabled)
# ============================================================

upload_col, result_col, chat_col = st.columns([1, 1.1, 0.9])


# ------------------------------------------------------------
# 1. UPLOAD
# ------------------------------------------------------------

with upload_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(panel_header_html(1, "Upload Plant Image"), unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-description">Upload a clear image of a plant leaf. '
        'Crop Doctor will analyze it using the trained CNN disease detection model.</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a plant image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded plant image", width="stretch")
        if st.button("✕ Remove image", key="remove_image"):
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# 2. CNN PREDICTION (functional)
# ------------------------------------------------------------

with result_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    if image is None:
        st.markdown(panel_header_html(2, "Prediction Results"), unsafe_allow_html=True)
        st.markdown(
            '<div class="panel-description">CNN-based disease classification using the '
            'trained Crop Doctor baseline model.</div>',
            unsafe_allow_html=True,
        )
        st.info("Upload a plant image to see the CNN prediction.")

    else:
        with st.spinner("Analyzing plant image..."):
            model, idx_to_class, transform, device = load_crop_doctor()

            results = predict_image(
                image=image,
                model=model,
                transform=transform,
                idx_to_class=idx_to_class,
                device=device,
                top_k=10,
            )

        top_result = results[0]
        top_class_full = format_class_name(top_result["class"])
        top_probability = top_result["probability"]

        # split "Crop — Disease" into two lines if possible
        if " — " in top_class_full:
            crop_part, disease_part = top_class_full.split(" — ", 1)
            display_name = f"{crop_part} — {disease_part}"
        else:
            display_name = top_class_full

        diagnosis_key = (display_name, round(top_probability, 3))
        st.session_state["diagnosis"] = {
            "crop": crop_part if " — " in top_class_full else "Unknown crop",
            "disease": disease_part if " — " in top_class_full else top_class_full,
            "confidence": top_probability,
            "display": display_name,
        }

        if st.session_state["chat_diagnosis_key"] != diagnosis_key:
            st.session_state["chat_diagnosis_key"] = diagnosis_key
            st.session_state["chat_messages"] = []
            st.session_state["chat_interaction_id"] = None

        st.markdown(
            panel_header_html(2, "Prediction Results", confidence_badge_html(top_probability)),
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="prediction-name">{display_name}</div>
                <div class="conf-row">
                    <span class="conf-label">Confidence score</span>
                    <span class="conf-value">{top_probability:.1%}</span>
                </div>
                <div class="bar-track">
                    <div class="bar-fill-red" style="width:{top_probability * 100:.1f}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-size:13px; font-weight:700; color:#34473d; margin-bottom:8px;">'
            'Top 5 Predictions</div>',
            unsafe_allow_html=True,
        )

        top5 = results[:5]
        rest = results[5:10]

        for result in top5:
            class_name = format_class_name(result["class"])
            probability = result["probability"]
            st.markdown(
                f"""
                <div class="top5-row">
                    <div class="top5-label-row">
                        <span class="top5-name">{class_name}</span>
                        <span class="top5-pct">{probability:.1%}</span>
                    </div>
                    <div class="bar-track-light">
                        <div class="bar-fill-green" style="width:{probability * 100:.1f}%;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if rest:
            with st.expander("View All Predictions (Top 10)"):
                for result in rest:
                    class_name = format_class_name(result["class"])
                    probability = result["probability"]
                    st.markdown(
                        f"""
                        <div class="top5-row">
                            <div class="top5-label-row">
                                <span class="top5-name">{class_name}</span>
                                <span class="top5-pct">{probability:.1%}</span>
                            </div>
                            <div class="bar-track-light">
                                <div class="bar-fill-green" style="width:{probability * 100:.1f}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# 3. ASK CROP DOCTOR (live chat, grounded in the current diagnosis)
# ------------------------------------------------------------

with chat_col:
    diagnosis = st.session_state["diagnosis"]

    if diagnosis is None:
        st.markdown('<div class="panel disabled-panel">', unsafe_allow_html=True)
        st.markdown(panel_header_html(3, "Ask Crop Doctor"), unsafe_allow_html=True)
        st.markdown(
            """
            <div class="chat-bubble">
                <div class="chat-avatar">🌱</div>
                <div class="chat-text">
                    Hi! I'm Crop Doctor. Diagnose a plant image first, and I'll help you
                    work through it.
                </div>
            </div>
            <div class="coming-soon-tag">🔒 Upload &amp; analyze an image to start chatting</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(panel_header_html(3, "Ask Crop Doctor"), unsafe_allow_html=True)
        st.markdown(
            f'<div class="panel-description">Grounded in: <b>{diagnosis["disease"]}</b> '
            f'&nbsp;·&nbsp; replying in {st.session_state["language"]}</div>',
            unsafe_allow_html=True,
        )

        if not st.session_state["chat_messages"]:
            st.markdown(
                """
                <div class="chat-bubble">
                    <div class="chat-avatar">🌱</div>
                    <div class="chat-text">
                        Hi! I'm Crop Doctor. Ask me anything about this diagnosis —
                        treatment, yield impact, or how to prevent it next season.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        chat_box = st.container(height=260)
        with chat_box:
            for msg in st.session_state["chat_messages"]:
                if msg["role"] == "assistant":
                    st.markdown(
                        f"""
                        <div class="chat-bubble">
                            <div class="chat-avatar">🌱</div>
                            <div class="chat-text">{msg["content"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="chat-bubble-user">
                            <div class="chat-bubble-user-text">
                                {msg["content"]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        user_question = st.chat_input("Type your question...")

        if user_question:
            st.session_state["chat_messages"].append({"role": "user", "content": user_question})
            try:
                with st.spinner("Crop Doctor is thinking..."):
                    reply, new_interaction_id = chat_with_expert(
                        crop=diagnosis["crop"],
                        disease=diagnosis["disease"],
                        confidence=diagnosis["confidence"],
                        language=st.session_state["language"],
                        user_message=user_question,
                        previous_interaction_id=st.session_state["chat_interaction_id"],
                    )
                st.session_state["chat_interaction_id"] = new_interaction_id
                st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
            except LLMConfigError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                render_llm_error(exc)
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)


# ============================================================
# SECOND ROW: Disease Info | Treatment | Prevention (disabled)
# ============================================================

info_col, treatment_col, prevention_col = st.columns(3)

diagnosis = st.session_state["diagnosis"]
language = st.session_state["language"]

explanation = None
review_meta = None
explanation_error = None

if diagnosis is not None:
    try:
        cache_key = (
            "explanation",
            diagnosis["crop"],
            diagnosis["disease"],
            round(diagnosis["confidence"], 3),
            language,
        )
        with st.spinner("🌿 Crop Doctor AI is analyzing the diagnosis... (may pause briefly if a free-tier rate limit is hit)"):
            explanation, review_meta = cached_llm_text(
                cache_key,
                lambda: generate_diagnosis_explanation(
                    crop=diagnosis["crop"],
                    disease=diagnosis["disease"],
                    confidence=diagnosis["confidence"],
                    language=language,
                ),
            )
    except LLMConfigError as exc:
        explanation_error = exc
    except Exception as exc:  # noqa: BLE001
        explanation_error = exc


# ------------------------------------------------------------
# 4. DISEASE INFORMATION (live — one structured call for the row)
# ------------------------------------------------------------

with info_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(panel_header_html(4, "🌿 Disease Information"), unsafe_allow_html=True)

    if diagnosis is None:
        st.markdown(
            '<div class="panel-description">Understand the detected disease, symptoms, '
            'causes and affected crops.</div>',
            unsafe_allow_html=True,
        )
        st.info("Diagnose a plant image to see its summary here.")
    elif explanation_error:
        render_llm_error(explanation_error)
    else:
        st.markdown(dark_text_html(explanation.diagnosis, bold=True), unsafe_allow_html=True)
        st.markdown(dark_text_html(explanation.disease_meaning), unsafe_allow_html=True)
        st.markdown(dark_text_html("Symptoms", bold=True), unsafe_allow_html=True)
        st.markdown(bullet_list_html(explanation.symptoms), unsafe_allow_html=True)
        st.markdown(dark_text_html("Possible causes", bold=True), unsafe_allow_html=True)
        st.markdown(bullet_list_html(explanation.possible_causes), unsafe_allow_html=True)
        st.markdown(
            f"<div class='explain-caption'>⚠️ {explanation.uncertainty}</div>",
            unsafe_allow_html=True,
        )
        render_review_meta(review_meta)

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# 5. TREATMENT RECOMMENDATIONS (live)
# ------------------------------------------------------------

with treatment_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(panel_header_html(5, "🛠 Treatment Recommendations"), unsafe_allow_html=True)

    if diagnosis is None:
        st.markdown(
            '<div class="panel-description">Receive AI-powered recommendations for '
            'treatment and disease management.</div>',
            unsafe_allow_html=True,
        )
        st.info("Diagnose a plant image to get an action plan.")
    elif explanation_error:
        render_llm_error(explanation_error)
    else:
        for i, step in enumerate(explanation.immediate_steps, start=1):
            st.markdown(dark_text_html(f"{i}. {step}"), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------
# 6. PREVENTION TIPS (now live too — it's the same structured call)
# ------------------------------------------------------------

with prevention_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(panel_header_html(6, "🛡 Prevention Tips"), unsafe_allow_html=True)

    if diagnosis is None:
        st.markdown(
            '<div class="panel-description">Learn preventive practices and strategies for '
            'healthier crops.</div>',
            unsafe_allow_html=True,
        )
        st.info("Diagnose a plant image to see prevention tips.")
    elif explanation_error:
        render_llm_error(explanation_error)
    else:
        st.markdown(bullet_list_html(explanation.prevention), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="app-footer">
        <div>Crop Doctor v1.0 &nbsp;|&nbsp; 🌿 AI for a Healthier Planet</div>
        <div>About &nbsp;·&nbsp; Privacy &nbsp;·&nbsp; Terms &nbsp;·&nbsp; Contact</div>
    </div>
    """,
    unsafe_allow_html=True,
)