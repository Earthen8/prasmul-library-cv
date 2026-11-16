import streamlit as st


def get_theme_variables(is_dark_mode: bool) -> str:
    if is_dark_mode:
        return """
        :root {
            --bg-page: #121417;
            --bg-surface: #1B1F24;
            --text-primary: #F3F4F6;
            --text-secondary: #A1A8B3;
            --border-strong: #6B7280;
            --accent-primary: #1D4ED8;
            --accent-text: #FFFFFF;
            --focus-ring: #93C5FD;
            --alert-bg: #450A0A;
            --alert-text: #FCA5A5;
            --alert-border: #F87171;
            --ok-bg: #052E16;
            --ok-text: #86EFAC;
        }
        """
    return """
    :root {
        --bg-page: #FAF8F5;
        --bg-surface: #F1EFEA;
        --text-primary: #18181B;
        --text-secondary: #4A4A52;
        --border-strong: #71717A;
        --accent-primary: #1E3A5F;
        --accent-text: #FFFFFF;
        --focus-ring: #1E3A5F;
        --alert-bg: #FEF2F2;
        --alert-text: #991B1B;
        --alert-border: #DC2626;
        --ok-bg: #F0FDF4;
        --ok-text: #166534;
    }
    """


def get_base_layout_css() -> str:
    return """
    .stApp {
        background-color: var(--bg-page) !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stSidebar"] {
        background-color: var(--bg-surface) !important;
        border-right: 1px solid var(--border-strong) !important;
    }
    h1, h2, h3, p, label, span, div {
        color: var(--text-primary);
    }
    h1 { font-size: clamp(1.45rem, 2.2vw, 2rem) !important; font-weight: 700 !important; }
    h2 { font-size: clamp(1.15rem, 1.6vw, 1.45rem) !important; font-weight: 600 !important; }
    h3 { font-size: clamp(1rem, 1.3vw, 1.15rem) !important; font-weight: 600 !important; }
    .panel-box {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-strong);
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .status-banner-alert {
        background-color: var(--alert-bg);
        color: var(--alert-text) !important;
        border: 2px solid var(--alert-border);
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        background-color: var(--bg-surface);
        font-variant-numeric: tabular-nums;
    }
    th, td {
        border: 1px solid var(--border-strong) !important;
        padding: 0.55rem 0.75rem !important;
        text-align: left !important;
        color: var(--text-primary) !important;
    }
    th {
        background-color: var(--bg-page) !important;
        font-weight: 600 !important;
    }
    """


def get_accessibility_and_responsive_css() -> str:
    return """
    a[href], button, summary, select, label,
    [role="button"], [role="tab"], [role="switch"],
    [data-baseweb="select"] {
        cursor: pointer !important;
    }
    button:disabled, [disabled] {
        cursor: not-allowed !important;
        opacity: 0.65 !important;
    }
    .stButton > button {
        min-height: 44px !important;
        padding: 0.5rem 1rem !important;
        background-color: var(--accent-primary) !important;
        color: var(--accent-text) !important;
        border: 2px solid var(--border-strong) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        margin-bottom: 0.35rem !important;
    }
    .stButton > button:hover {
        filter: brightness(1.12);
        border-color: var(--text-primary) !important;
    }
    .stButton > button:active {
        transform: translateY(1px);
    }
    button:focus-visible, a:focus-visible, input:focus-visible,
    select:focus-visible, [tabindex]:focus-visible {
        outline: 3px solid var(--focus-ring) !important;
        outline-offset: 2px !important;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-bottom: 1rem;
        font-variant-numeric: tabular-nums;
    }
    .metric-item {
        background-color: var(--bg-surface);
        border: 1px solid var(--border-strong);
        border-radius: 6px;
        padding: 0.75rem 1rem;
    }
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary) !important;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: clamp(1.25rem, 2vw, 1.6rem);
        font-weight: 700;
        color: var(--text-primary) !important;
    }
    @media (max-width: 1024px) {
        .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 640px) {
        .metric-grid { grid-template-columns: 1fr; }
        .block-container { padding: 1rem 0.75rem !important; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            transition-duration: 0.01ms !important;
        }
    }
    """


def apply_custom_styles(is_dark_mode: bool) -> None:
    css = (
        "<style>"
        + get_theme_variables(is_dark_mode)
        + get_base_layout_css()
        + get_accessibility_and_responsive_css()
        + "</style>"
    )
    st.markdown(css, unsafe_allow_html=True)
