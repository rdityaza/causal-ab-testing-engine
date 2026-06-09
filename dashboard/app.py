import sys
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.db_pipeline import fetch_from_db
from src.ab_testing import check_srm, calculate_significance
from src.causal_engine import run_causal_impact

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Causal A/B Testing Engine",
    page_icon="https://cdn-icons-png.flaticon.com/512/3090/3090011.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── GLOBAL STYLES ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Font & base */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,1,0');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }

  /* Background */
  .stApp {
      background-color: #f8fafc;
      color: #0f172a;
  }

  /* Hide default Streamlit header & footer */
  #MainMenu, footer, header { visibility: hidden; }

  /* Custom top nav bar */
  .topbar {
      background: #ffffff;
      border-bottom: 1px solid #e2e8f0;
      padding: 16px 32px;
      margin: -4rem -4rem 2rem -4rem;
      display: flex;
      align-items: center;
      gap: 12px;
  }
  .topbar-icon { font-size: 1.5rem; }
  .topbar-title {
      font-size: 1.1rem;
      font-weight: 700;
      color: #0f172a;
      letter-spacing: -0.02em;
  }
  .topbar-sub {
      font-size: 0.75rem;
      color: #718096;
      margin-top: 2px;
  }
  .topbar-badge {
      margin-left: auto;
      background: #1e3a5f;
      color: #63b3ed;
      font-size: 0.7rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 20px;
      border: 1px solid #2b6cb0;
      font-family: 'JetBrains Mono', monospace;
  }

  /* Section header */
  .section-header {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 1rem;
  }
  .section-label {
      font-size: 0.65rem;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #4a5568;
  }
  .section-title {
      font-size: 1rem;
      font-weight: 700;
      color: #0f172a;
  }
  .section-divider {
      flex: 1;
      height: 1px;
      background: #e2e8f0;
  }

  /* Card */
  .card {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 10px;
      padding: 20px 24px;
      height: 100%;
  }
  .card-label {
      font-size: 0.7rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #718096;
      margin-bottom: 6px;
  }
  .card-value {
      font-size: 1.9rem;
      font-weight: 700;
      color: #0f172a;
      font-family: 'JetBrains Mono', monospace;
      line-height: 1;
      margin-bottom: 6px;
  }
  .card-value.positive { color: #68d391; }
  .card-value.negative { color: #fc8181; }
  .card-desc {
      font-size: 0.75rem;
      color: #718096;
      line-height: 1.5;
  }

  /* Status banner */
  .status-safe {
      background: #1c3a2a;
      border: 1px solid #2f6b4a;
      border-left: 4px solid #48bb78;
      border-radius: 8px;
      padding: 14px 18px;
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.85rem;
      color: #9ae6b4;
  }
  .status-danger {
      background: #3b1e1e;
      border: 1px solid #7b3333;
      border-left: 4px solid #fc8181;
      border-radius: 8px;
      padding: 14px 18px;
      display: flex;
      align-items: center;
      gap: 10px;
      font-size: 0.85rem;
      color: #feb2b2;
  }
  .status-icon { font-size: 1.1rem; }

  /* Significance chip */
  .chip-sig {
      display: inline-block;
      background: #1c3a5f;
      color: #63b3ed;
      border: 1px solid #2b6cb0;
      font-size: 0.7rem;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 20px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
  }
  .chip-neutral {
      display: inline-block;
      background: #e2e8f0;
      color: #a0aec0;
      border: 1px solid #4a5568;
      font-size: 0.7rem;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 20px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
  }

  /* Override metric style */
  [data-testid="stMetric"] {
      background: transparent !important;
  }
  [data-testid="stMetricValue"] {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.6rem !important;
      color: #0f172a !important;
  }
  [data-testid="stMetricLabel"] {
      font-size: 0.7rem !important;
      color: #718096 !important;
      text-transform: uppercase;
      letter-spacing: 0.08em;
  }

  /* Plot background override */
  .stPlotlyChart, .element-container iframe { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── TOP BAR ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <span class="material-symbols-rounded" style="font-size: 28px;">show_chart</span>
    <div>
        <div class="topbar-title">Causal A/B Testing Engine</div>
        <div class="topbar-sub">One-Click Checkout · Conversion Rate & Revenue Analysis</div>
    </div>
    <div class="topbar-badge">LIVE</div>
</div>
""", unsafe_allow_html=True)


# ── HELPER: section header ────────────────────────────────────────────────────
def section_header(step: str, title: str):
    st.markdown(f"""
    <div class="section-header">
        <span class="section-label">{step}</span>
        <span class="section-title">{title}</span>
        <div class="section-divider"></div>
    </div>
    """, unsafe_allow_html=True)


# ── HELPER: custom card ───────────────────────────────────────────────────────
def stat_card(label: str, value: str, desc: str, value_class: str = ""):
    st.markdown(f"""
    <div class="card">
        <div class="card-label">{label}</div>
        <div class="card-value {value_class}">{value}</div>
        <div class="card-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


# ── 1. FETCH DATA ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    query = "SELECT * FROM experiment_logs ORDER BY timestamp"
    return fetch_from_db(query)

with st.spinner("Menarik data dari PostgreSQL..."):
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Gagal terhubung ke database. Error: {e}")
        st.stop()


# ── 2. GUARDRAILS CHECK ───────────────────────────────────────────────────────
section_header("STEP 01", "Guardrails · Sample Ratio Mismatch Check")

is_safe = check_srm(df)
ctrl_count = len(df[df['group'] == 'Control'])
treat_count = len(df[df['group'] == 'Treatment'])
total = ctrl_count + treat_count

col_a, col_b, col_c = st.columns([1, 1, 2])
with col_a:
    stat_card("Control", f"{ctrl_count:,}", f"{ctrl_count/total*100:.1f}% dari total traffic")
with col_b:
    stat_card("Treatment", f"{treat_count:,}", f"{treat_count/total*100:.1f}% dari total traffic")
with col_c:
    if is_safe:
        st.markdown("""
        <div class="status-safe">
            <span class="material-symbols-rounded" style="color: #48bb78;">check_circle</span>
            <div>
                <strong>Traffic Split Valid (50:50)</strong><br>
                Tidak terdeteksi Sample Ratio Mismatch. Data aman untuk dianalisis lebih lanjut.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-danger">
            <span class="material-symbols-rounded" style="color: #fc8181;">error</span>
            <div>
                <strong>SRM Terdeteksi!</strong><br>
                Ada indikasi bug pada sistem routing eksperimen. Analisis dihentikan.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

st.markdown("<br>", unsafe_allow_html=True)


# ── 3. STANDARD A/B TESTING ───────────────────────────────────────────────────
section_header("STEP 02", "Standard A/B Testing · Statistical Significance")

results = calculate_significance(df)

conv = results['conversion_rate']
rev  = results['revenue']

c1, c2, c3, c4 = st.columns(4)

# Conversion Rate
with c1:
    ctrl_cvr  = conv.get('control_rate', 0) * 100
    treat_cvr = conv.get('treatment_rate', 0) * 100
    stat_card("Control CVR", f"{ctrl_cvr:.2f}%", "Baseline conversion rate")

with c2:
    delta_cvr = treat_cvr - ctrl_cvr
    v_class   = "positive" if delta_cvr > 0 else "negative"
    stat_card(
        "Treatment CVR",
        f"{treat_cvr:.2f}%",
        f"Δ {delta_cvr:+.2f}pp vs Control",
        v_class
    )

# Revenue
with c3:
    ctrl_rev  = rev.get('control_mean', 0)
    treat_rev = rev.get('treatment_mean', 0)
    stat_card("Avg Revenue · Control", f"Rp {ctrl_rev:,.0f}", "Per user, baseline")

with c4:
    delta_rev = treat_rev - ctrl_rev
    v_class   = "positive" if delta_rev > 0 else "negative"
    stat_card(
        "Avg Revenue · Treatment",
        f"Rp {treat_rev:,.0f}",
        f"Δ Rp {delta_rev:+,.0f} vs Control",
        v_class
    )

# Significance summary row
st.markdown("<br>", unsafe_allow_html=True)
s1, s2 = st.columns(2)

with s1:
    sig_label = "SIGNIFICANT" if conv['is_significant'] else "NOT SIGNIFICANT"
    chip_cls  = "chip-sig" if conv['is_significant'] else "chip-neutral"
    insight   = "Fitur meningkatkan proporsi checkout secara signifikan." if conv['is_significant'] else "Belum ada bukti kuat perbedaan conversion."
    st.markdown(f"""
    <div class="card">
        <div class="card-label">Z-Test · Conversion Rate</div>
        <div style="margin: 8px 0;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:1.3rem; color:#0f172a;">
                p = {conv['p_value']:.4f}
            </span>
            &nbsp;&nbsp;<span class="{chip_cls}">{sig_label}</span>
        </div>
        <div class="card-desc">{insight}</div>
    </div>
    """, unsafe_allow_html=True)

with s2:
    sig_label = "SIGNIFICANT" if rev['is_significant'] else "NOT SIGNIFICANT"
    chip_cls  = "chip-sig" if rev['is_significant'] else "chip-neutral"
    insight   = "Fitur meningkatkan rata-rata belanja per user secara signifikan." if rev['is_significant'] else "Belum ada bukti kuat perbedaan revenue."
    st.markdown(f"""
    <div class="card">
        <div class="card-label">Welch's T-Test · Revenue per User</div>
        <div style="margin: 8px 0;">
            <span style="font-family:'JetBrains Mono',monospace; font-size:1.3rem; color:#0f172a;">
                p = {rev['p_value']:.4f}
            </span>
            &nbsp;&nbsp;<span class="{chip_cls}">{sig_label}</span>
        </div>
        <div class="card-desc">{insight}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── 4. CAUSAL IMPACT ──────────────────────────────────────────────────────────
section_header("STEP 03", "Causal Impact · Counterfactual Analysis")

df['date'] = pd.to_datetime(df['timestamp']).dt.normalize()
daily_rev  = df.groupby(['date', 'group'])['revenue'].sum().unstack().fillna(0)
df_causal  = pd.DataFrame({'y': daily_rev['Treatment'], 'X': daily_rev['Control']})

dates_list  = df_causal.index.strftime('%Y-%m-%d').tolist()
pre_period  = [dates_list[0],  dates_list[14]]
post_period = [dates_list[15], dates_list[-1]]

with st.spinner("Membangun counterfactual model..."):
    ci_model, ci_metrics = run_causal_impact(df_causal, pre_period, post_period)

# Metric cards
m1, m2, m3 = st.columns(3)
with m1:
    stat_card("Total Causal Effect", f"Rp {ci_metrics['absolute_effect']:,.0f}",
              "Estimasi tambahan revenue akibat fitur",
              "positive" if ci_metrics['absolute_effect'] > 0 else "negative")
with m2:
    stat_card("Relative Lift", f"{ci_metrics['relative_effect_pct']:.2f}%",
              "Kenaikan relatif vs counterfactual baseline",
              "positive" if ci_metrics['relative_effect_pct'] > 0 else "negative")
with m3:
    p = ci_metrics['p_value']
    stat_card("P-Value (Bayesian)", f"{p:.4f}",
              "Kausal signifikan." if p < 0.05 else "Efek kausal belum signifikan.")

st.markdown("<br>", unsafe_allow_html=True)

# ── CHART ─────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

# 1. Bersihkan "ingatan" matplotlib biar grafiknya gak numpuk/double
plt.clf()
plt.close('all')

# 2. Gambar plot bawaan pycausalimpact
ci_model.plot()

# 3. Tangkap grafiknya
fig = plt.gcf()

# 4. (Opsional) Bikin background grafiknya jadi putih bersih biar nyatu sama UI Light Mode
fig.patch.set_facecolor('#ffffff')
for ax in fig.get_axes():
    ax.set_facecolor('#ffffff')

# 5. Tampilkan ke Streamlit
st.pyplot(fig, use_container_width=True)

st.markdown("""
<div style="text-align:center; margin-top:2rem; color:#718096; font-size:0.75rem;">
    Pre-period: 15 hari pertama &nbsp;·&nbsp; Post-period: sisa durasi eksperimen &nbsp;·&nbsp;
    Model: Bayesian Structural Time Series (CausalImpact)
</div>
""", unsafe_allow_html=True)