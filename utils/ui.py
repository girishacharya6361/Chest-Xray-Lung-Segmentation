"""Shared Streamlit styling and helpers (white / light-grey, navy + teal, rounded cards)."""
import streamlit as st

NAVY, TEAL = "#0B2545", "#13A89E"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ================= GLOBAL TEXT ================= */
html, body, [class*="css"], .stMarkdown, button, input, textarea {{
    font-family: 'Inter','Segoe UI',system-ui,sans-serif !important;
}}

.stApp {{
    background: linear-gradient(180deg,#ffffff 0%,#f3f5f8 100%);
}}

html, body, .stApp, [data-testid="stAppViewContainer"] {{
    color: {NAVY} !important;
}}

p, li, label, span, div, small {{
    color: inherit;
}}

h1, h2, h3, h4, h5, h6 {{
    color: {NAVY} !important;
    letter-spacing: -0.01em;
}}

/* ================= SIDEBAR ================= */
section[data-testid="stSidebar"] {{
    background: #f1f5f9 !important;
    border-right: 1px solid #dbe4ee !important;
}}

section[data-testid="stSidebar"] * {{
    color: {NAVY} !important;
}}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
    color: {NAVY} !important;
}}

section[data-testid="stSidebar"] button {{
    color: {NAVY} !important;
    background: transparent !important;
    border-radius: 10px !important;
}}

section[data-testid="stSidebar"] button:hover {{
    background: #d9f3ef !important;
    border-color: {TEAL} !important;
}}

section[data-testid="stSidebar"] a {{
    color: #087f78 !important;
}}

/* ================= HERO ================= */
.hero {{
    background: linear-gradient(120deg,{NAVY} 0%,#12426f 60%,{TEAL} 140%);
    color: #ffffff !important;
    padding: 28px 32px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(11,37,69,.18);
    margin-bottom: 22px;
}}

.hero h1 {{
    color: #ffffff !important;
    font-size: 1.9rem;
    margin: 0 0 6px 0;
    font-weight: 800;
}}

.hero p {{
    color: #e8f1f8 !important;
    margin: 0;
    font-size: 1.02rem;
}}

/* ================= CARDS ================= */
.card {{
    background: #ffffff !important;
    border: 1px solid #e6eaf0;
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 4px 16px rgba(11,37,69,.06);
    margin-bottom: 16px;
    height: 100%;
}}

.card h4 {{
    color: {NAVY} !important;
    margin: 0 0 8px 0;
    font-weight: 700;
}}

.card p, .card li {{
    color: #3b4a5e !important;
    font-size: .95rem;
}}

/* ================= KPI ================= */
.kpi {{
    background: #ffffff !important;
    border: 1px solid #e6eaf0;
    border-left: 5px solid {TEAL};
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 4px 16px rgba(11,37,69,.06);
}}

.kpi .v {{
    font-size: 1.7rem;
    font-weight: 800;
    color: {NAVY} !important;
    line-height: 1.1;
}}

.kpi .l {{
    font-size: .78rem;
    color: #52647a !important;
    text-transform: uppercase;
    letter-spacing: .06em;
    font-weight: 600;
}}

.kpi .h {{
    font-size: .8rem;
    color: #52647a !important;
    margin-top: 4px;
}}

/* ================= STEP CARDS ================= */
.step {{
    background: #ffffff !important;
    border: 1px solid #e6eaf0;
    border-radius: 14px;
    padding: 12px 14px;
    font-size: .88rem;
    color: {NAVY} !important;
    box-shadow: 0 3px 10px rgba(11,37,69,.05);
    height: 100%;
}}

.step b {{
    color: {TEAL} !important;
}}

/* ================= BADGES ================= */
.badge {{
    display: inline-block;
    padding: 3px 11px;
    border-radius: 999px;
    font-size: .75rem;
    font-weight: 700;
}}

.badge.ok {{
    background: #dcf5f2 !important;
    color: #0b7a72 !important;
}}

.badge.no {{
    background: #fdeaea !important;
    color: #b3261e !important;
}}

/* ================= BUTTONS ================= */
.stButton>button,
.stDownloadButton>button {{
    border-radius: 12px;
    font-weight: 600;
    border: 1px solid {NAVY} !important;
    background: {NAVY} !important;
    color: #ffffff !important;
    padding: .5rem 1.1rem;
}}

.stButton>button p,
.stDownloadButton>button p {{
    color: #ffffff !important;
}}

.stButton>button:hover,
.stDownloadButton>button:hover {{
    background: {TEAL} !important;
    border-color: {TEAL} !important;
    color: #ffffff !important;
}}

.stButton>button:disabled {{
    background: #e2e8f0 !important;
    color: #64748b !important;
    border-color: #cbd5e1 !important;
}}

/* ================= INPUTS ================= */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
.stDateInput input,
.stTimeInput input {{
    background: #ffffff !important;
    color: {NAVY} !important;
    border: 1px solid #cbd5e1 !important;
}}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {{
    color: #64748b !important;
    opacity: 1 !important;
}}

[data-testid="stSelectbox"] div,
[data-testid="stMultiSelect"] div {{
    color: {NAVY};
}}

[data-baseweb="select"] {{
    background: #ffffff !important;
    color: {NAVY} !important;
}}

[data-baseweb="input"] {{
    background: #ffffff !important;
    color: {NAVY} !important;
}}

/* ================= CHECKBOX & RADIO ================= */
.stCheckbox label,
.stRadio label,
.stToggle label {{
    color: {NAVY} !important;
}}

/* ================= TABS ================= */
.stTabs [data-baseweb="tab"] {{
    color: {NAVY} !important;
    font-weight: 600;
}}

.stTabs [aria-selected="true"] {{
    color: #087f78 !important;
    border-bottom-color: {TEAL} !important;
}}

/* ================= TABLES ================= */
[data-testid="stDataFrame"],
[data-testid="stTable"] {{
    background: #ffffff !important;
    color: {NAVY} !important;
}}

[data-testid="stDataFrame"] * {{
    color: {NAVY} !important;
}}

/* ================= METRICS ================= */
[data-testid="stMetric"] {{
    background: #ffffff !important;
    border: 1px solid #e6eaf0;
    padding: 12px;
    border-radius: 12px;
}}

[data-testid="stMetricLabel"] {{
    color: #52647a !important;
}}

[data-testid="stMetricValue"] {{
    color: {NAVY} !important;
}}

[data-testid="stMetricDelta"] {{
    color: #087f78 !important;
}}

/* ================= ALERTS ================= */
[data-testid="stAlert"] {{
    border-radius: 12px;
}}

[data-testid="stAlert"] p {{
    color: {NAVY} !important;
}}

/* ================= FILE UPLOADER ================= */
[data-testid="stFileUploaderDropzone"] {{
    border: 2px dashed {TEAL} !important;
    border-radius: 16px;
    background: #f9fffe !important;
}}

[data-testid="stFileUploaderDropzone"] * {{
    color: {NAVY} !important;
}}

[data-testid="stFileUploaderDropzone"] button {{
    background: #ffffff !important;
    color: {NAVY} !important;
    border: 1px solid {TEAL} !important;
}}

/* ================= IMAGES ================= */
[data-testid="stImage"] img {{
    border-radius: 12px;
}}

/* ================= EXPANDERS ================= */
[data-testid="stExpander"] {{
    background: #ffffff !important;
    border: 1px solid #dbe4ee !important;
    border-radius: 12px !important;
}}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {{
    color: {NAVY} !important;
}}

/* ================= CODE ================= */
[data-testid="stCode"] {{
    border-radius: 12px;
}}

/* ================= CAPTIONS ================= */
.stCaption,
[data-testid="stCaptionContainer"] {{
    color: #52647a !important;
}}

/* ================= LINKS ================= */
a {{
    color: #087f78 !important;
}}

/* ================= RESPONSIVE ================= */
@media (max-width: 800px) {{
    .hero h1 {{
        font-size: 1.4rem;
    }}
}}
</style>
"""


def setup(title, icon="🫁"):
    st.set_page_config(
        page_title=f"{title} | Chest X-Ray Segmentation",
        page_icon=icon,
        layout="wide"
    )
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle=""):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )


def card(title, body_html):
    st.markdown(
        f'<div class="card"><h4>{title}</h4>{body_html}</div>',
        unsafe_allow_html=True
    )


def kpi(label, value, hint=""):
    st.markdown(
        f'<div class="kpi"><div class="l">{label}</div>'
        f'<div class="v">{value}</div><div class="h">{hint}</div></div>',
        unsafe_allow_html=True
    )


def badge(ok, yes="Ready", no="Missing"):
    return f'<span class="badge {"ok" if ok else "no"}">{yes if ok else no}</span>'


def need_image():
    """Stop the page with a friendly message when no X-ray has been uploaded."""
    if "gray" not in st.session_state:
        st.info(
            "No X-ray uploaded yet. Go to **Upload Xray** in the sidebar, "
            "upload an image and press *Process*."
        )
        st.stop()
    return st.session_state["gray"]


def sidebar_brand():
    st.sidebar.markdown(
        f"<div style='font-weight:800;color:{NAVY};font-size:1.05rem'>🫁 Chest X-Ray<br>"
        f"<span style='color:{TEAL}'>Enhancement & Segmentation</span></div>",
        unsafe_allow_html=True
    )
    if "gray" in st.session_state:
        st.sidebar.caption(
            f"Current image: **{st.session_state.get('filename', 'x-ray')}**"
        )