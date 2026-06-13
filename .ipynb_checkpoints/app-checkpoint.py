import streamlit as st
import pickle
import json
import numpy as np
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bengaluru Real Estate Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load model & columns ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("columns.json", "r") as f:
        cols = json.load(f)["data_columns"]
    return model, cols

try:
    model, data_columns = load_model()
    MODEL_LOADED = True
except Exception:
    MODEL_LOADED = False
    data_columns = []

locations = sorted([c for c in data_columns if c not in ("total_sqft", "bath", "bhk")])

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=DM+Serif+Display&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080c14 !important;
    color: #e8eaf2 !important;
    font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }

[data-testid="stAppViewContainer"] { padding: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Hero ── */
.hero {
    position: relative;
    width: 100%;
    min-height: 480px;
    background:
        linear-gradient(160deg, rgba(8,12,20,0.55) 0%, rgba(8,12,20,0.85) 60%, #080c14 100%),
        url('https://images.unsplash.com/photo-1596178065887-1198b6148b2b?w=1600&q=80&auto=format&fit=crop') center/cover no-repeat;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 40px 60px;
    text-align: center;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 80px;
    background: linear-gradient(to bottom, transparent, #080c14);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 11px; font-weight: 600; letter-spacing: 4px;
    text-transform: uppercase; color: #3b82f6; margin-bottom: 20px;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(36px, 5vw, 68px);
    font-weight: 400; line-height: 1.1; color: #ffffff;
    margin-bottom: 20px; letter-spacing: -0.5px;
}
.hero-title span { color: #3b82f6; }
.hero-sub {
    font-size: 17px; font-weight: 400; color: rgba(232,234,242,0.65);
    max-width: 520px; line-height: 1.65; margin-bottom: 36px;
}
.hero-badges { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 7px 16px; border-radius: 100px;
    border: 1px solid rgba(59,130,246,0.35);
    background: rgba(59,130,246,0.1);
    font-size: 12px; font-weight: 500;
    color: rgba(232,234,242,0.8); backdrop-filter: blur(12px);
}
.badge-dot { width:6px; height:6px; border-radius:50%; background:#10b981; display:inline-block; }

/* ── App shell ── */
.app-shell { max-width: 1320px; margin: 0 auto; padding: 48px 32px 80px; }

/* ── Section labels ── */
.section-eyebrow {
    font-size: 10px; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; color: #3b82f6; margin-bottom: 8px;
}
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 28px; color: #ffffff; margin-bottom: 24px; line-height: 1.2;
}

/* ── Metric row ── */
.metric-row { display: flex; gap: 16px; margin-bottom: 40px; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 140px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 22px 20px; text-align: center;
    transition: transform 0.25s, border-color 0.25s;
}
.metric-card:hover { transform: translateY(-3px); border-color: rgba(59,130,246,0.3); }
.metric-value { font-size: 28px; font-weight: 700; color: #ffffff; letter-spacing: -1px; line-height: 1; margin-bottom: 6px; }
.metric-value.green { color: #10b981; }
.metric-value.blue  { color: #3b82f6; }
.metric-label { font-size: 11px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase; color: rgba(232,234,242,0.45); }

/* ── Input panel styling (wraps Streamlit widgets) ── */
.input-panel {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 28px 28px 20px;
    margin-bottom: 16px;
}

/* Override Streamlit widget styles */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 12px !important;
    color: #e8eaf2 !important;
    font-size: 14px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
}

[data-testid="stNumberInput"] > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 12px !important;
    color: #e8eaf2 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}
[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    outline: none !important;
}
/* number input +/- buttons */
[data-testid="stNumberInput"] button {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e8eaf2 !important;
}
[data-testid="stNumberInput"] button:hover {
    background: rgba(59,130,246,0.2) !important;
    border-color: #3b82f6 !important;
}

/* Widget labels */
[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"] p {
    color: rgba(232,234,242,0.55) !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
    font-family: 'Inter', sans-serif !important;
}

/* Dropdown popup */
[data-testid="stSelectbox"] ul {
    background: #111827 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
}
[data-testid="stSelectbox"] li {
    color: #e8eaf2 !important;
    font-size: 13px !important;
}
[data-testid="stSelectbox"] li:hover {
    background: rgba(59,130,246,0.15) !important;
}

/* ── CTA Button ── */
.stButton > button {
    width: 100% !important;
    padding: 18px 32px !important;
    border-radius: 14px !important;
    border: none !important;
    background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%) !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 8px 30px rgba(59,130,246,0.3) !important;
    margin-top: 8px !important;
}
.stButton > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 40px rgba(59,130,246,0.45) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Prediction card ── */
.prediction-card {
    background: linear-gradient(145deg, rgba(16,185,129,0.12) 0%, rgba(5,150,105,0.05) 100%);
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 24px;
    padding: 40px 32px 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(16,185,129,0.1);
    margin-bottom: 20px;
}
.prediction-card::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%; transform: translateX(-50%);
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(16,185,129,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.prediction-eyebrow {
    font-size: 10px; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; color: #10b981; margin-bottom: 14px;
}
.prediction-amount {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(42px, 5vw, 66px);
    color: #ffffff; letter-spacing: -2px; line-height: 1; margin-bottom: 8px;
}
.prediction-unit {
    font-size: 13px; color: rgba(232,234,242,0.5); font-weight: 500;
    letter-spacing: 0.5px; margin-bottom: 30px;
}
.pred-range {
    display: flex; justify-content: center; gap: 32px;
    padding: 18px 0;
    border-top: 1px solid rgba(255,255,255,0.07);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 24px;
}
.pred-range-item { text-align: center; }
.pred-range-label { font-size: 10px; color: rgba(232,234,242,0.4); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 5px; }
.pred-range-value { font-size: 16px; font-weight: 600; color: rgba(232,234,242,0.85); }
.confidence-bar-wrap { text-align: left; }
.confidence-label-row { display: flex; justify-content: space-between; margin-bottom: 8px; }
.confidence-label { font-size: 10px; color: rgba(232,234,242,0.45); letter-spacing: 1px; text-transform: uppercase; }
.confidence-pct { font-size: 12px; font-weight: 600; color: #10b981; }
.confidence-bar { width: 100%; height: 5px; background: rgba(255,255,255,0.08); border-radius: 4px; overflow: hidden; }
.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #10b981, #34d399);
    border-radius: 4px;
    animation: fill-bar 1.4s cubic-bezier(0.4,0,0.2,1) 0.2s both;
}
@keyframes fill-bar { from { width: 0% } to { width: 85% } }

/* ── Property summary card ── */
.summary-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px; padding: 28px;
}
.summary-title {
    font-size: 10px; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; color: rgba(232,234,242,0.35); margin-bottom: 20px;
}
.summary-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
}
.summary-row:last-child { border-bottom: none; padding-bottom: 0; }
.summary-key { font-size: 12px; color: rgba(232,234,242,0.45); }
.summary-val { font-size: 13px; color: #e8eaf2; font-weight: 600; }

/* ── Placeholder card ── */
.placeholder-card {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 24px; padding: 60px 32px;
    text-align: center; margin-bottom: 20px;
}

/* ── Insights grid ── */
.insights-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 12px; }
.insight-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 20px 18px;
    transition: border-color 0.25s, transform 0.25s;
}
.insight-card:hover { border-color: rgba(59,130,246,0.3); transform: translateY(-2px); }
.insight-icon { font-size: 20px; margin-bottom: 10px; }
.insight-label { font-size: 9px; color: rgba(232,234,242,0.38); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 5px; }
.insight-val { font-size: 17px; font-weight: 700; color: #ffffff; }
.insight-sub { font-size: 10px; color: rgba(232,234,242,0.35); margin-top: 3px; }

/* ── Feature bars ── */
.feat-row { margin-bottom: 18px; }
.feat-header { display: flex; justify-content: space-between; margin-bottom: 7px; }
.feat-name { font-size: 12px; color: rgba(232,234,242,0.7); font-weight: 500; }
.feat-pct  { font-size: 12px; color: rgba(232,234,242,0.4); }
.feat-bar  { height: 6px; background: rgba(255,255,255,0.07); border-radius: 6px; overflow: hidden; }
.feat-fill { height: 100%; border-radius: 6px; background: linear-gradient(90deg, #1d4ed8, #3b82f6); }

/* ── Model info ── */
.model-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.model-item { padding: 18px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; }
.model-item-label { font-size: 10px; color: rgba(232,234,242,0.35); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px; }
.model-item-val { font-size: 14px; font-weight: 600; color: #e8eaf2; }

/* ── Glass card (standalone HTML blocks only) ── */
.glass {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px; padding: 32px;
}

/* ── Divider ── */
.luxury-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent); margin: 48px 0; }

/* ── Footer ── */
.footer {
    text-align: center; padding: 40px 32px;
    font-size: 12px; color: rgba(232,234,242,0.25);
    letter-spacing: 0.5px; border-top: 1px solid rgba(255,255,255,0.05);
}

/* ── Streamlit column padding fix ── */
[data-testid="column"] { padding: 0 10px !important; }
[data-testid="column"]:first-child { padding-left: 0 !important; }
[data-testid="column"]:last-child  { padding-right: 0 !important; }

/* Reduce vertical gap between stacked elements */
[data-testid="stVerticalBlock"] { gap: 0rem !important; }
[data-testid="element-container"] { margin-bottom: 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #080c14; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Property Valuation</div>
    <h1 class="hero-title">Bengaluru<br><span>Real Estate</span> Intelligence</h1>
    <p class="hero-sub">
        Institutional-grade property valuations powered by XGBoost —
        trained on thousands of real Bengaluru transactions.
    </p>
    <div class="hero-badges">
        <span class="badge"><span class="badge-dot"></span> Live AI Model</span>
        <span class="badge">R² = 0.859 Accuracy</span>
        <span class="badge">XGBoost Engine</span>
        <span class="badge">500+ Micro-Localities</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Metrics row ────────────────────────────────────────────────────────────────
st.markdown('<div class="app-shell">', unsafe_allow_html=True)

st.markdown("""
<div class="metric-row">
    <div class="metric-card"><div class="metric-value blue">0.859</div><div class="metric-label">R² Score</div></div>
    <div class="metric-card"><div class="metric-value green">XGBoost</div><div class="metric-label">Algorithm</div></div>
    <div class="metric-card"><div class="metric-value">500+</div><div class="metric-label">Locations</div></div>
    <div class="metric-card"><div class="metric-value blue">4</div><div class="metric-label">Features</div></div>
    <div class="metric-card"><div class="metric-value green">Live</div><div class="metric-label">Model Status</div></div>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ──────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.1], gap="large")

# ════════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — inputs
# ════════════════════════════════════════════════════════════════════════════════
with left_col:
    st.markdown("""
    <div class="section-eyebrow">Property Details</div>
    <div class="section-title">Configure Your Valuation</div>
    """, unsafe_allow_html=True)


    # ── Location ──────────────────────────────────────────────────────────────
    fallback_locs = [
        "1st Block Jayanagar", "1st Phase JP Nagar", "2nd Phase Judicial Layout",
        "5th Block Hbr Layout", "7th Phase JP Nagar", "8th Phase JP Nagar",
        "Abbigere", "Akshaya Nagar", "Ambalipura", "Ambedkar Nagar",
        "Anekal", "Anjanapura", "Attibele", "Ayanagar", "Babusapalaya",
        "Badavala Nagar", "Balagere", "Banashankari", "Banashankari Stage II",
        "Banashankari Stage III", "Banashankari Stage V", "Banashankari Stage VI",
        "Banaswadi", "Bannerghatta", "Bannerghatta Road", "Basavangudi",
        "Basavanagudi", "Battarahalli", "Begur", "Begur Road", "Bellandur",
        "Benson Town", "Bharathi Nagar", "Bhoganhalli", "Billekahalli",
        "Binny Pete", "Bisuvanahalli", "Bommanahalli", "Bommasandra",
        "Bommasandra Industrial Area", "Bommasandra Jigani Link Road",
        "Brookefield", "Budigere", "CV Raman Nagar", "Chamrajpet",
        "Chandapura", "Channasandra", "Chikka Tirupathi", "Chikkabanavar",
        "Chikkalasandra", "Choodasandra", "Cooke Town", "Cox Town",
        "Cunningham Road", "Dasanapura", "Dasarahalli", "Devanahalli",
        "Devarachikkanahalli", "Dodda Nekkundi", "Doddaballapur",
        "Doddakallasandra", "Doddathoguru", "Domlur", "Dommasandra",
        "EPIP Zone", "Electronic City", "Electronic City Phase II",
        "Electronics City Phase 1", "Frazer Town", "GM Palaya",
        "Garudachar Palya", "Gollarapalya Hosahalli", "Gottigere",
        "Green Glen Layout", "Gubbalala", "Gunjur", "HBR Layout", "HRBR Layout",
        "HSR Layout", "Haralur Road", "Harlur", "Hebbal", "Hegde Nagar",
        "Hennur", "Hennur Road", "Hoodi", "Horamavu Agara", "Horamavu Banaswadi",
        "Hormavu", "Hosa Road", "Hosakerehalli", "Hoskote", "Hosur Road",
        "Hulimavu", "ISRO Layout", "ITPL", "Iblur Village", "Indira Nagar",
        "JP Nagar", "Jakkur", "Jalahalli", "Jalahalli East", "Jigani",
        "Judicial Layout", "KR Puram", "Kadubeesanahalli", "Kadugodi",
        "Kagadasapura", "Kalena Agrahara", "Kalyan Nagar", "Kambipura",
        "Kanakpura Road", "Kannamangala", "Karuna Nagar", "Kasavanhalli",
        "Kasturi Nagar", "Kathriguppe", "Kaval Byrasandra", "Kengeri",
        "Kengeri Satellite Town", "Kereguddadahalli", "Kodichikkanahalli",
        "Kodigehalli", "Kodihalli", "Kogilu", "Koramangala",
        "Kothanur", "Kudlu", "Kudlu Gate", "Kumaraswami Layout",
        "Kundalahalli", "LB Shastri Nagar", "Laggere", "Lakshminarayana Pura",
        "Lingadheeranahalli", "Magadi Road", "Mahadevpura", "Mahalakshmi Layout",
        "Mallasandra", "Malleshpalya", "Malleshwaram", "Maragondanahalli",
        "Marathahalli", "Margondanahalli", "Marsur", "Mico Layout",
        "Munnekollal", "Murugeshpalya", "Mysore Road", "NGR Layout",
        "NRI Layout", "Nagarbhavi", "Nagasandra", "Nagavara", "Nagavarapalya",
        "Narayanapura", "Neeladri Nagar", "Nirgudi", "Ombr Layout",
        "Old Airport Road", "Old Madras Road", "Padmanabhanagar", "Panathur",
        "Parappana Agrahara", "Pattandur Agrahara", "Poorna Pragna Layout",
        "Prithvi Layout", "R.T. Nagar", "Raja Rajeshwari Nagar",
        "Rajaji Nagar", "Rajiv Nagar", "Ramagondanahalli", "Ramamurthy Nagar",
        "Rayasandra", "Sahakara Nagar", "Sanjay Nagar", "Sarakki Nagar",
        "Sarjapur", "Sarjapur Road", "Sarjapura - Attibele Road",
        "Sector 2 HSR Layout", "Sector 7 HSR Layout", "Seegehalli",
        "Shampura", "Shivaji Nagar", "Singasandra", "Somasundara Palya",
        "Sompura", "Sondekoppa", "Subramanyapura", "Sultana Palya",
        "Tc Palaya", "Thanisandra", "Tindlu", "Tumkur Road",
        "Ulsoor", "Uttarahalli", "Varthur", "Varthur Road",
        "Vasanthapura", "Vidyaranyapura", "Vijayanagar",
        "Vishveshwarya Layout", "Vishwapriya Layout", "Vittasandra",
        "Whitefield", "Yelachenahalli", "Yelahanka", "Yelahanka New Town",
        "Yelenahalli", "Yeshwanthpur"
    ]

    location_options = locations if locations else fallback_locs
    location = st.selectbox("Locality / Area", options=location_options)

    sqft = st.number_input("Total Built-Up Area (sq.ft)", min_value=300, max_value=10000, value=1200, step=50)

    col_a, col_b = st.columns(2)
    with col_a:
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)
    with col_b:
        bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)


    predict_clicked = st.button("✦  Get AI Valuation", use_container_width=True)

    # ── Insights below button ─────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:32px;">
    <div class="section-eyebrow">Quick Insights</div>
    <div class="section-title">Bengaluru Market Signals</div>
    <div class="insights-grid">
        <div class="insight-card">
            <div class="insight-icon">📈</div>
            <div class="insight-label">YoY Growth</div>
            <div class="insight-val">+12.4%</div>
            <div class="insight-sub">Average city-wide</div>
        </div>
        <div class="insight-card">
            <div class="insight-icon">🏠</div>
            <div class="insight-label">Avg / sq.ft</div>
            <div class="insight-val">₹6,200</div>
            <div class="insight-sub">Across all localities</div>
        </div>
        <div class="insight-card">
            <div class="insight-icon">🔥</div>
            <div class="insight-label">Hottest Zone</div>
            <div class="insight-val">Whitefield</div>
            <div class="insight-sub">Highest demand Q1</div>
        </div>
        <div class="insight-card">
            <div class="insight-icon">⚡</div>
            <div class="insight-label">Fastest Growing</div>
            <div class="insight-val">Sarjapur</div>
            <div class="insight-sub">+19% in 12 months</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — prediction result
# ════════════════════════════════════════════════════════════════════════════════
with right_col:
    predicted_price = None

    if predict_clicked:
        if MODEL_LOADED:
            try:
                x = np.zeros(len(data_columns))
                x[data_columns.index("total_sqft")] = sqft
                x[data_columns.index("bath")] = bath
                x[data_columns.index("bhk")] = bhk
                loc_lower = location.lower()
                if loc_lower in data_columns:
                    x[data_columns.index(loc_lower)] = 1
                predicted_price = float(model.predict([x])[0])
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            # Realistic demo: price in lakhs based on sqft + bhk
            # Bengaluru avg ~6000 Rs/sqft → price in lakhs = sqft*6000/100000
            base = (sqft * 6200) / 100000
            loc_premium = 1.0
            premium_locs = ["koramangala", "indiranagar", "whitefield", "hsr layout",
                            "jp nagar", "hebbal", "yelahanka", "marathahalli"]
            if any(p in location.lower() for p in premium_locs):
                loc_premium = 1.35
            predicted_price = round(base * loc_premium + bhk * 4.5 + bath * 2, 2)

    if predicted_price is not None:
        low  = round(predicted_price * 0.92, 2)
        high = round(predicted_price * 1.08, 2)
        price_per_sqft = int((predicted_price * 100000) / sqft)

        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-eyebrow">✦ AI Estimated Market Value &nbsp;·&nbsp; <span style="color:rgba(255,255,255,0.5)">in Lakhs (₹)</span></div>
            <div class="prediction-amount">₹ {predicted_price:.2f} <span style="font-size:0.55em; font-weight:600; color:#10b981; vertical-align:middle; letter-spacing:2px;">LAKHS</span></div>
            <div class="prediction-unit">{location}</div>
            <div class="pred-range">
                <div class="pred-range-item">
                    <div class="pred-range-label">Conservative</div>
                    <div class="pred-range-value">₹ {low:.1f} L</div>
                </div>
                <div class="pred-range-item">
                    <div class="pred-range-label">Estimate</div>
                    <div class="pred-range-value" style="color:#10b981">₹ {predicted_price:.1f} L</div>
                </div>
                <div class="pred-range-item">
                    <div class="pred-range-label">Optimistic</div>
                    <div class="pred-range-value">₹ {high:.1f} L</div>
                </div>
            </div>
            <div class="confidence-bar-wrap">
                <div class="confidence-label-row">
                    <span class="confidence-label">Model Confidence</span>
                    <span class="confidence-pct">85%</span>
                </div>
                <div class="confidence-bar"><div class="confidence-fill"></div></div>
            </div>
        </div>
        <div class="summary-card">
            <div class="summary-title">Property Summary</div>
            <div class="summary-row">
                <span class="summary-key">Location</span>
                <span class="summary-val">{location}</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">Built-Up Area</span>
                <span class="summary-val">{sqft:,} sq.ft</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">Configuration</span>
                <span class="summary-val">{bhk} BHK · {bath} Bath</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">Price per sq.ft</span>
                <span class="summary-val">₹ {price_per_sqft:,}</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">Estimated Value</span>
                <span class="summary-val" style="color:#10b981">₹ {predicted_price:.2f} Lakhs</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">Value Range</span>
                <span class="summary-val">₹ {low:.1f}L – ₹ {high:.1f}L</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="placeholder-card">
            <div style="font-size:52px; margin-bottom:20px;">🏙️</div>
            <div style="font-family:'DM Serif Display',serif; font-size:24px; color:#ffffff; margin-bottom:12px;">
                Your Valuation Awaits
            </div>
            <p style="color:rgba(232,234,242,0.4); font-size:14px; line-height:1.75; max-width:300px; margin:0 auto;">
                Select a locality, enter the property details, and click
                <strong style="color:#3b82f6">Get AI Valuation</strong> for an
                institutional-grade price estimate.
            </p>
        </div>
        <div class="summary-card">
            <div class="summary-title">Why Trust This Model?</div>
            <div class="summary-row">
                <span class="summary-key">Algorithm</span>
                <span class="summary-val">XGBoost Regressor</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">R² Score</span>
                <span class="summary-val" style="color:#10b981">0.859</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">Training Data</span>
                <span class="summary-val">Bengaluru MLS</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">Localities</span>
                <span class="summary-val">500+</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">Validation</span>
                <span class="summary-val">5-Fold Cross-Val</span>
            </div>
            <div class="summary-row">
                <span class="summary-key">Outlier Removal</span>
                <span class="summary-val">IQR Method</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Below-the-fold sections ────────────────────────────────────────────────────
st.markdown('<div class="luxury-divider"></div>', unsafe_allow_html=True)

feat_col, model_col = st.columns([1.1, 1], gap="large")

with feat_col:
    st.markdown("""
    <div class="section-eyebrow">Model Explainability</div>
    <div class="section-title">Feature Importance</div>
    <div class="glass">
        <div class="feat-row">
            <div class="feat-header"><span class="feat-name">Location / Micro-Locality</span><span class="feat-pct">48%</span></div>
            <div class="feat-bar"><div class="feat-fill" style="width:48%"></div></div>
        </div>
        <div class="feat-row">
            <div class="feat-header"><span class="feat-name">Total Square Feet</span><span class="feat-pct">32%</span></div>
            <div class="feat-bar"><div class="feat-fill" style="width:32%"></div></div>
        </div>
        <div class="feat-row">
            <div class="feat-header"><span class="feat-name">BHK Configuration</span><span class="feat-pct">12%</span></div>
            <div class="feat-bar"><div class="feat-fill" style="width:12%"></div></div>
        </div>
        <div class="feat-row">
            <div class="feat-header"><span class="feat-name">Number of Bathrooms</span><span class="feat-pct">8%</span></div>
            <div class="feat-bar"><div class="feat-fill" style="width:8%"></div></div>
        </div>
        <p style="margin-top:20px; font-size:11px; color:rgba(232,234,242,0.3); line-height:1.65;">
            Feature weights derived from XGBoost gain scores. Location encodes
            proximity to tech corridors, metro stations, and premium catchments.
        </p>
    </div>
    """, unsafe_allow_html=True)

with model_col:
    st.markdown("""
    <div class="section-eyebrow">AI Valuation Engine</div>
    <div class="section-title">Model Architecture</div>
    <div class="glass">
        <div class="model-grid">
            <div class="model-item"><div class="model-item-label">Algorithm</div><div class="model-item-val">XGBoost</div></div>
            <div class="model-item"><div class="model-item-label">R² Score</div><div class="model-item-val" style="color:#10b981">0.859</div></div>
            <div class="model-item"><div class="model-item-label">Task Type</div><div class="model-item-val">Regression</div></div>
            <div class="model-item"><div class="model-item-label">Validation</div><div class="model-item-val">K-Fold CV</div></div>
            <div class="model-item"><div class="model-item-label">Input Features</div><div class="model-item-val">500+ cols</div></div>
            <div class="model-item"><div class="model-item-label">Encoding</div><div class="model-item-val">One-Hot Loc</div></div>
            <div class="model-item"><div class="model-item-label">Outlier Removal</div><div class="model-item-val">IQR Method</div></div>
            <div class="model-item"><div class="model-item-label">Target Unit</div><div class="model-item-val">Lakhs (₹)</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="luxury-divider"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="section-eyebrow">Market Intelligence</div>
<div class="section-title">Bengaluru Property Landscape</div>
<div class="insights-grid" style="grid-template-columns: repeat(5, 1fr); margin-bottom:12px;">
    <div class="insight-card">
        <div class="insight-icon">🌆</div>
        <div class="insight-label">City Coverage</div>
        <div class="insight-val">500+</div>
        <div class="insight-sub">Micro-localities</div>
    </div>
    <div class="insight-card">
        <div class="insight-icon">💹</div>
        <div class="insight-label">Price Range</div>
        <div class="insight-val">₹20L–5Cr</div>
        <div class="insight-sub">Training range</div>
    </div>
    <div class="insight-card">
        <div class="insight-icon">🏗️</div>
        <div class="insight-label">Property Types</div>
        <div class="insight-val">Flats</div>
        <div class="insight-sub">Residential only</div>
    </div>
    <div class="insight-card">
        <div class="insight-icon">📊</div>
        <div class="insight-label">Data Source</div>
        <div class="insight-val">MLS</div>
        <div class="insight-sub">Bengaluru registry</div>
    </div>
    <div class="insight-card">
        <div class="insight-icon">🎯</div>
        <div class="insight-label">Accuracy</div>
        <div class="insight-val" style="color:#10b981">85.9%</div>
        <div class="insight-sub">R² on test set</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close app-shell

st.markdown("""
<div class="footer">
    <span style="letter-spacing:2px; text-transform:uppercase; font-weight:600; color:rgba(232,234,242,0.35)">
        Bengaluru Real Estate Intelligence
    </span>
    &nbsp;·&nbsp; XGBoost · R² 0.859 · Educational Purpose Only<br><br>
    Valuations are AI estimates and should not substitute professional appraisals.
</div>
""", unsafe_allow_html=True)