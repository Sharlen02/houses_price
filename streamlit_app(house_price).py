import streamlit as st
import pandas as pd
from snowflake.snowpark.context import get_active_session
from snowflake.ml.registry import Registry
from sklearn.preprocessing import StandardScaler

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Reset & base */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d0f14 !important;
    color: #e8e4dd !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(212,163,97,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(97,149,212,0.08) 0%, transparent 60%),
        #0d0f14 !important;
}

/* Hide default header/footer */
[data-testid="stHeader"], footer { display: none !important; }
[data-testid="block-container"] { padding: 2rem 3rem !important; max-width: 1200px; margin: 0 auto; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.7rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #d4a361;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 400;
    line-height: 1.1;
    color: #f0ebe2;
    margin: 0 0 0.5rem;
}
.hero-title em {
    font-style: italic;
    color: #d4a361;
}
.hero-sub {
    font-size: 0.95rem;
    color: #7a7670;
    font-weight: 300;
    letter-spacing: 0.02em;
}

/* ── Divider ── */
.gold-line {
    width: 48px; height: 1px;
    background: linear-gradient(90deg, transparent, #d4a361, transparent);
    margin: 1.5rem auto;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #d4a361;
    margin-bottom: 1.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(212,163,97,0.2);
}

/* ── Card panels ── */
.card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
    transition: border-color 0.3s;
}
.card:hover { border-color: rgba(212,163,97,0.25); }

/* ── Streamlit widget overrides ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select,
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e8e4dd !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(212,163,97,0.5) !important;
    box-shadow: 0 0 0 2px rgba(212,163,97,0.15) !important;
}

/* Slider track */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #d4a361 !important;
    border-color: #d4a361 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"] > div {
    background: #d4a361 !important;
}

/* Labels */
label, [data-testid="stWidgetLabel"] p {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    color: #9a9590 !important;
    text-transform: uppercase !important;
}

/* ── CTA Button ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #d4a361 0%, #c4913f 100%) !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 2rem !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 20px rgba(212,163,97,0.25) !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(212,163,97,0.4) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, rgba(212,163,97,0.08) 0%, rgba(212,163,97,0.03) 100%);
    border: 1px solid rgba(212,163,97,0.35);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
    animation: fadeUp 0.5s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #d4a361;
    margin-bottom: 0.6rem;
}
.result-price {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.8rem, 6vw, 4.5rem);
    color: #f0ebe2;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.result-currency {
    font-size: 1.8rem;
    color: #d4a361;
    vertical-align: super;
    margin-right: 0.2rem;
}

/* ── Market bar ── */
.market-bar-wrap {
    margin: 1.5rem 0 0.5rem;
}
.market-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.7rem;
    color: #5a5650;
    margin-bottom: 0.5rem;
    letter-spacing: 0.05em;
}
.market-bar-bg {
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 99px;
    overflow: hidden;
}
.market-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #d4a361, #f0c070);
    border-radius: 99px;
    transition: width 1s cubic-bezier(0.4,0,0.2,1);
}
.market-position {
    font-size: 0.72rem;
    color: #7a7670;
    text-align: center;
    margin-top: 0.6rem;
}

/* ── Metrics row ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1.5rem;
}
.metric-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #e8e4dd;
}
.metric-lbl {
    font-size: 0.65rem;
    color: #5a5650;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary {
    color: #7a7670 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    font-size: 0.68rem;
    color: #3a3830;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.app-footer span { color: #d4a361; }

/* ── Success/info override ── */
[data-testid="stAlert"] {
    background: rgba(212,163,97,0.08) !important;
    border: 1px solid rgba(212,163,97,0.3) !important;
    border-radius: 10px !important;
    color: #e8e4dd !important;
}
</style>
""", unsafe_allow_html=True)


# ─── CONSTANTS ───────────────────────────────────────────────────────────────
RAW_FEATURES = [
    'AREA', 'BEDROOMS', 'BATHROOMS', 'STORIES', 'MAINROAD',
    'GUESTROOM', 'BASEMENT', 'HOTWATERHEATING', 'AIRCONDITIONING',
    'PARKING', 'PREFAREA', 'FURNISHINGSTATUS'
]
FEATURE_NAMES = RAW_FEATURES + ['AREA_PER_BEDROOM', 'COMFORT_SCORE', 'AREA_X_STORIES']
MIN_PRICE, MAX_PRICE = 87_500, 665_000


# ─── SESSION & MODEL ─────────────────────────────────────────────────────────
session = get_active_session()

@st.cache_resource
def load_scaler_and_model(model_alias: str = "prod"):
    pdf = session.table("HOUSE_PRICE.ML.HOUSE_PRICES").to_pandas()

    binary_cols = ["MAINROAD", "GUESTROOM", "BASEMENT",
                   "HOTWATERHEATING", "AIRCONDITIONING", "PREFAREA"]
    for col in binary_cols:
        pdf[col] = pdf[col].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})

    pdf["FURNISHINGSTATUS"] = pdf["FURNISHINGSTATUS"].astype(str).str.strip().str.lower().map(
        {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}
    )

    scaler = StandardScaler()
    scaler.fit(pdf[RAW_FEATURES])

    reg   = Registry(session=session, database_name="HOUSE_PRICE", schema_name="ML")
    model = reg.get_model("HOUSE_PRICE_XGBOOST").version(model_alias)
    return scaler, model

scaler, model = load_scaler_and_model("prod")


# ─── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Snowflake ML · XGBoost Model : Réalisé par D'ALMEIDA Morènikè Sharlen, Bientakonne KARAMBIRI et Stephen AGGEY</div>
    <h1 class="hero-title">Estimation du Prix<br><em>d'une Maison</em></h1>
    <p class="hero-sub">Renseignez les caractéristiques pour obtenir une estimation instantanée</p>
    <div class="gold-line"></div>
</div>
""", unsafe_allow_html=True)


# ─── FORM ────────────────────────────────────────────────────────────────────
col_a, col_b, col_c = st.columns([1, 1, 1], gap="medium")

with col_a:
    with st.container(border=True):
        st.markdown('<div class="section-label">📐 Superficie & Structure</div>', unsafe_allow_html=True)
        area      = st.number_input("Surface (m²)",  min_value=30,  max_value=400, value=150, step=5)
        bedrooms  = st.slider("Chambres",            min_value=1,   max_value=6,   value=3)
        bathrooms = st.slider("Salles de bain",      min_value=1,   max_value=4,   value=2)
        stories   = st.slider("Étages",              min_value=1,   max_value=4,   value=2)
        parking   = st.slider("Places de parking",   min_value=0,   max_value=3,   value=1)

with col_b:
    with st.container(border=True):
        st.markdown('<div class="section-label">🏗️ Équipements</div>', unsafe_allow_html=True)
        mainroad        = st.selectbox("Route principale",  ["yes", "no"])
        guestroom       = st.selectbox("Chambre d'amis",    ["no",  "yes"])
        basement        = st.selectbox("Sous-sol",           ["no",  "yes"])
        hotwaterheating = st.selectbox("Chauffe-eau",        ["no",  "yes"])
        airconditioning = st.selectbox("Climatisation",      ["no",  "yes"])

with col_c:
    with st.container(border=True):
        st.markdown('<div class="section-label">✨ Prestations</div>', unsafe_allow_html=True)
        prefarea   = st.selectbox("Zone privilégiée", ["no", "yes"])
        furnishing = st.selectbox("Ameublement", ["furnished", "semi-furnished", "unfurnished"])
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">📊 Résumé</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.82rem; color:#7a7670; line-height:2;">
            <b style="color:#e8e4dd">{area} m²</b> · {bedrooms} ch. · {bathrooms} sdb<br>
            {stories} étage(s) · {parking} parking(s)<br>
            Clim : <b style="color:#e8e4dd">{'✓' if airconditioning=='yes' else '✗'}</b> &nbsp;
            Cave : <b style="color:#e8e4dd">{'✓' if basement=='yes' else '✗'}</b><br>
            <span style="color:#d4a361">{furnishing.replace('-', ' ').title()}</span>
        </div>
        """, unsafe_allow_html=True)


# ─── ESTIMATE BUTTON ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run = st.button("💡 Estimer le prix", use_container_width=True)


# ─── INFERENCE ───────────────────────────────────────────────────────────────
def yn(v): return 1 if v == "yes" else 0
furn_map = {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}

if run:
    with st.spinner("Calcul en cours…"):

        # 1. Raw DataFrame (12 features)
        input_raw = pd.DataFrame([{
            "AREA":             area,
            "BEDROOMS":         bedrooms,
            "BATHROOMS":        bathrooms,
            "STORIES":          stories,
            "MAINROAD":         yn(mainroad),
            "GUESTROOM":        yn(guestroom),
            "BASEMENT":         yn(basement),
            "HOTWATERHEATING":  yn(hotwaterheating),
            "AIRCONDITIONING":  yn(airconditioning),
            "PARKING":          parking,
            "PREFAREA":         yn(prefarea),
            "FURNISHINGSTATUS": furn_map[furnishing]
        }], columns=RAW_FEATURES)

        # 2. Scale 12 raw features
        input_scaled = pd.DataFrame(
            scaler.transform(input_raw),
            columns=RAW_FEATURES
        )

        # 3. Feature engineering APRÈS scaling
        input_scaled["AREA_PER_BEDROOM"] = input_scaled["AREA"] / (input_scaled["BEDROOMS"] + 1)
        input_scaled["COMFORT_SCORE"]    = (
            input_scaled["BASEMENT"] + input_scaled["HOTWATERHEATING"]
            + input_scaled["AIRCONDITIONING"] + input_scaled["GUESTROOM"]
        )
        input_scaled["AREA_X_STORIES"]   = input_scaled["AREA"] * input_scaled["STORIES"]

        # 4. Réordonner (15 features)
        input_scaled = input_scaled[FEATURE_NAMES]

        # 5. Predict
        prediction = model.run(input_scaled, function_name="predict")
        price = float(prediction.values.flatten()[0])

    # ── Result display ──
    pct      = min(max((price - MIN_PRICE) / (MAX_PRICE - MIN_PRICE), 0), 1)
    pct_pct  = round(pct * 100)
    price_m2 = round(price / area)

    # Prix principal — simple HTML (une seule div, pas d'imbrication profonde)
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimation XGBoost</div>
        <div class="result-price"><span class="result-currency">€</span>{price:,.0f}</div>
        <div style="font-size:0.8rem;color:#7a7670;margin-top:0.4rem;">
            soit environ <strong style="color:#d4a361">€ {price_m2:,} / m²</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Barre marché — composants natifs Streamlit
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;font-size:0.7rem;
                color:#5a5650;margin:1.2rem 0 0.4rem;letter-spacing:0.05em;">
        <span>Min · €{MIN_PRICE:,}</span>
        <span>Positionnement dans la fourchette du marché ({pct_pct}%)</span>
        <span>Max · €{MAX_PRICE:,}</span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(pct)

    # Métriques — st.metric natif
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🛏 Chambres",      bedrooms)
    m2.metric("📐 Surface",       f"{area} m²")
    m3.metric("🏗 Étages",        stories)
    m4.metric("📍 Prix / m²",     f"€ {price_m2:,}")

    with st.expander("📋 Détail des caractéristiques saisies"):
        display_df = input_raw.copy()
        display_df.index = ["Maison estimée"]
        st.dataframe(display_df, use_container_width=True)

    st.balloons()


# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Modèle · <span>XGBoost</span> &nbsp;·&nbsp;
    Registry · <span>Snowflake ML</span> &nbsp;·&nbsp;
    Dataset · <span>1 090 maisons</span>
</div>
""", unsafe_allow_html=True)