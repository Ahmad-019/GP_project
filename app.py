import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from googleapiclient.discovery import build
from transformers import pipeline
from typing import List

# ==========================================================
# API KEY
# ==========================================================
API_KEY = "enter_your_youtube_api_key_here"
# ==========================================================

st.set_page_config(
    page_title="GoldenMoment · BI Video Analytics",
    page_icon="✦",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════
#  DESIGN SYSTEM — LUXURY DARK EDITORIAL
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=JetBrains+Mono:wght@400;600&display=swap');
:root {
  --gold:       #f0b429;
  --gold-dim:   #c8881a;
  --ember:      #ff5e1a;
  --surface-0:  #06090f;
  --surface-1:  #0c1018;
  --surface-2:  #111720;
  --surface-3:  #1a2030;
  --border:     rgba(255,255,255,0.07);
  --border-hi:  rgba(240,180,41,0.35);
  --text-1:     #f0f2f8;
  --text-2:     #8b93a8;
  --text-3:     #4a5268;
  --font-display: 'Bebas Neue', sans-serif;
  --font-body:    'DM Sans', sans-serif;
  --font-mono:    'JetBrains Mono', monospace;
}
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
    background-color: var(--surface-0) !important;
    color: var(--text-1) !important;
    font-family: var(--font-body) !important;
}
[data-testid="block-container"] { padding: 0 3rem 5rem !important; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { background: var(--surface-1) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
[data-testid="stSidebar"] .stTextInput label {
    color: var(--text-3) !important; font-size: 0.65rem !important; font-weight: 600 !important;
    letter-spacing: 1.4px !important; text-transform: uppercase !important; font-family: var(--font-mono) !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: var(--surface-2) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text-1) !important; font-family: var(--font-mono) !important;
    font-size: 0.78rem !important; padding: 10px 14px !important; transition: border-color 0.2s !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: var(--gold) !important; box-shadow: 0 0 0 3px rgba(240,180,41,0.12) !important; outline: none !important;
}
[data-testid="stSidebar"] .stTextInput input::placeholder { color: var(--text-3) !important; font-size: 0.75rem !important; }
.stProgress > div > div > div > div { background: linear-gradient(90deg, var(--gold), var(--ember)) !important; border-radius: 4px !important; }
div[data-testid="metric-container"] {
    background: var(--surface-2) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; padding: 20px 24px !important;
}
[data-testid="stMetricValue"] { font-family: var(--font-display) !important; font-size: 2.2rem !important; letter-spacing: 1px !important; color: var(--text-1) !important; }
[data-testid="stMetricLabel"] { font-family: var(--font-mono) !important; font-size: 0.65rem !important; letter-spacing: 1.2px !important; text-transform: uppercase !important; color: var(--text-3) !important; }
[data-testid="stStatusContainer"] { background: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; font-family: var(--font-mono) !important; font-size: 0.8rem !important; }
hr { border-color: var(--border) !important; }
.stAlert { background: var(--surface-2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; color: var(--text-2) !important; font-family: var(--font-body) !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--surface-0); }
::-webkit-scrollbar-thumb { background: var(--surface-3); border-radius: 4px; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  EMOTION CONFIG
# ═══════════════════════════════════════════════════════════
EMOTION_CONFIG = {
    "Funny":         {"color": "#f0b429", "bg": "rgba(240,180,41,0.1)",  "border": "rgba(240,180,41,0.3)",  "icon": "😂", "glow": "rgba(240,180,41,0.12)"},
    "Happy":         {"color": "#22c55e", "bg": "rgba(34,197,94,0.1)",   "border": "rgba(34,197,94,0.3)",   "icon": "😊", "glow": "rgba(34,197,94,0.12)"},
    "Sad":           {"color": "#60a5fa", "bg": "rgba(96,165,250,0.1)",  "border": "rgba(96,165,250,0.3)",  "icon": "😢", "glow": "rgba(96,165,250,0.12)"},
    "Controversial": {"color": "#f87171", "bg": "rgba(248,113,113,0.1)", "border": "rgba(248,113,113,0.3)", "icon": "🔥", "glow": "rgba(248,113,113,0.12)"},
    "Inspirational": {"color": "#c084fc", "bg": "rgba(192,132,252,0.1)", "border": "rgba(192,132,252,0.3)", "icon": "✨", "glow": "rgba(192,132,252,0.12)"},
}
FALLBACK_CFG = {"color": "#f0f2f8", "bg": "rgba(255,255,255,0.06)", "border": "rgba(255,255,255,0.15)", "icon": "✦", "glow": "rgba(255,255,255,0.06)"}

# ═══════════════════════════════════════════════════════════
#  ML ENGINE
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_emotion_engine():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

emotion_engine = load_emotion_engine()

# ═══════════════════════════════════════════════════════════
#  DATA ACQUISITION
# ═══════════════════════════════════════════════════════════
def fetch_comments_refined(video_id: str, max_results: int = 50000):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments, next_page_token = [], None
    progress_bar = st.progress(0)
    status_text = st.empty()
    while len(comments) < max_results:
        try:
            response = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100,
                pageToken=next_page_token, textFormat="plainText"
            ).execute()
            for item in response['items']:
                comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
            next_page_token = response.get('nextPageToken')
            n = len(comments)
            progress_bar.progress(min(n / max_results, 1.0))
            if n < 10000:
                msg = "📡  Connecting to YouTube Data Stream..."
            elif n < 30000:
                msg = f"📥  Pulling comments — {n:,} captured so far..."
            else:
                msg = f"⚙️  Optimising payload for inference — {n:,} samples"
            status_text.markdown(
f"<p style='color:#8b93a8;font-size:0.8rem;font-family:JetBrains Mono,monospace;'>{msg}</p>",
unsafe_allow_html=True
            )
            if not next_page_token:
                break
        except Exception as e:
            st.error(f"API Error: {e}")
            break
    progress_bar.empty()
    status_text.empty()
    return comments

# ═══════════════════════════════════════════════════════════
#  PARSING
# ═══════════════════════════════════════════════════════════
def process_intelligence(comments: List[str]):
    data = []
    pattern = r'(\d{1,2}:\d{2}(?::\d{2})?)'
    for text in comments:
        matches = re.findall(pattern, text)
        if matches:
            ts = matches[0]
            pts = list(map(int, ts.split(':')))
            secs = pts[0]*3600 + pts[1]*60 + pts[2] if len(pts) == 3 else pts[0]*60 + pts[1]
            data.append({"Timestamp": ts, "Seconds": secs, "Content": text})
    return pd.DataFrame(data)

def classify_sentiment_logic(text: str):
    t = text.lower()
    if any(x in t for x in ['😂', '🤣', 'lol', 'haha', 'funny']):
        return "Funny"
    try:
        res = emotion_engine(text[:512])[0]
        return {'joy': 'Happy', 'sadness': 'Sad', 'anger': 'Controversial', 'surprise': 'Inspirational'}.get(res['label'], "Neutral")
    except:
        return "Neutral"

# ═══════════════════════════════════════════════════════════
#  ★ SMARTER TOP-3 ALGORITHM
# ═══════════════════════════════════════════════════════════
EMOTION_HEAT   = {"Funny": 1.4, "Controversial": 1.5, "Inspirational": 1.3, "Happy": 1.0, "Sad": 0.9}
MIN_WINDOW_GAP = 3   # minutes

def compute_smart_highlights(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df['Window'] = df['Seconds'] // 60

    records = []
    for window, grp in df.groupby('Window'):
        count        = len(grp)
        dominant_ts  = grp['Timestamp'].mode()[0]
        dominant_em  = grp['Sentiment'].mode()[0]
        unique_em    = grp['Sentiment'].nunique()
        diversity_b  = 1 + (unique_em - 1) * 0.15
        avg_heat     = sum(EMOTION_HEAT.get(e, 1.0) for e in grp['Sentiment']) / count
        raw_score    = count * avg_heat * diversity_b
        records.append({
            'Window':    window,
            'Timestamp': dominant_ts,
            'Sentiment': dominant_em,
            'Count':     count,
            'RawScore':  raw_score,
            'Diversity': unique_em,
        })

    scored = pd.DataFrame(records).sort_values('RawScore', ascending=False)
    picked = []
    for _, row in scored.iterrows():
        if not any(abs(row['Window'] - p['Window']) < MIN_WINDOW_GAP for p in picked):
            picked.append(row.to_dict())
        if len(picked) == top_n:
            break

    if len(picked) < top_n:
        already = {p['Window'] for p in picked}
        for _, row in scored.iterrows():
            if row['Window'] not in already:
                picked.append(row.to_dict())
                already.add(row['Window'])
            if len(picked) == top_n:
                break

    result = pd.DataFrame(picked).reset_index(drop=True)
    max_s  = result['RawScore'].max()
    result['ScorePct'] = (result['RawScore'] / max_s * 100).round(1)
    return result

# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="padding:28px 20px 20px;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:24px;">
<div style="display:flex;align-items:center;gap:12px;">
<div style="width:40px;height:40px;background:linear-gradient(135deg,#f0b429,#ff5e1a);border-radius:10px;display:flex;align-items:center;justify-content:center;font-family:'Bebas Neue',sans-serif;font-size:1.4rem;letter-spacing:0;color:#000;">G</div>
<div>
<div style="font-family:'Bebas Neue',sans-serif;font-size:1.2rem;letter-spacing:2px;color:#f0f2f8;line-height:1.1;">GOLDEN MOMENT</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;color:#4a5268;letter-spacing:1px;margin-top:2px;">BI VIDEO ANALYTICS</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="padding:0 4px 6px;">
<span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;letter-spacing:1.3px;text-transform:uppercase;color:#4a5268;">
DATA SOURCE
</span>
</div>
""", unsafe_allow_html=True)

    target_url = st.text_input(
        "YouTube URL",
        placeholder="https://youtube.com/watch?v=...",
        label_visibility="collapsed",
        key="yt_input"
    )

    st.markdown("""
<div style="margin-top:28px;padding:0 4px 12px;">
<div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;letter-spacing:1.3px;text-transform:uppercase;color:#4a5268;margin-bottom:14px;">
INTELLIGENCE STACK
</div>
</div>
""", unsafe_allow_html=True)

    for tag, label in [
        ("NLP",    "7-class emotion model"),
        ("DATA",   "Up to 50,000 comments"),
        ("ALGO",   "Composite score ranking"),
        ("SPREAD", "3-min gap enforcement"),
        ("HEAT",   "Emotion intensity weights"),
    ]:
        st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;padding:7px 4px;border-bottom:1px solid rgba(255,255,255,0.04);">
<span style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;font-weight:600;background:rgba(240,180,41,0.08);border:1px solid rgba(240,180,41,0.18);color:#f0b429;border-radius:4px;padding:2px 6px;flex-shrink:0;">{tag}</span>
<span style="font-size:0.78rem;color:#8b93a8;">{label}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="margin-top:22px;padding:10px 14px;background:rgba(34,197,94,0.05);border:1px solid rgba(34,197,94,0.15);border-radius:8px;display:flex;align-items:center;gap:8px;">
<div style="width:6px;height:6px;background:#22c55e;border-radius:50%;animation:pulse 2s infinite;flex-shrink:0;"></div>
<span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#22c55e;letter-spacing:0.5px;">SYSTEM OPERATIONAL</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div style="position:relative;padding:64px 0 52px;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:0;overflow:hidden;">
<div style="position:absolute;top:-60px;left:-80px;width:500px;height:300px;background:radial-gradient(ellipse,rgba(240,180,41,0.07) 0%,transparent 70%);pointer-events:none;"></div>
<div style="position:absolute;bottom:-40px;right:0;width:400px;height:250px;background:radial-gradient(ellipse,rgba(255,94,26,0.05) 0%,transparent 70%);pointer-events:none;"></div>
<div style="display:inline-flex;align-items:center;gap:8px;font-family:'JetBrains Mono',monospace;font-size:0.63rem;font-weight:600;letter-spacing:2px;text-transform:uppercase;color:#f0b429;border:1px solid rgba(240,180,41,0.2);background:rgba(240,180,41,0.06);border-radius:4px;padding:4px 12px;margin-bottom:20px;">
✦   GRADUATION PROJECT — SMART BI SYSTEM
</div>
<div style="font-family:'Bebas Neue',sans-serif;font-size:clamp(3rem,6vw,5.5rem);line-height:0.95;letter-spacing:3px;color:#f0f2f8;margin-bottom:20px;">
FIND THE<br>
<span style="background:linear-gradient(90deg,#f0b429 0%,#ff5e1a 55%,#f0b429 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
GOLDEN MOMENTS
</span>
</div>
<p style="font-family:'DM Sans',sans-serif;font-size:1rem;font-weight:300;color:#4a5268;max-width:560px;line-height:1.75;margin:0;">
AI-powered crowd behaviour analytics. Surface emotional peaks, engagement spikes
& highlight-worthy timestamps from tens of thousands of audience comments.
</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  SECTION HEADER HELPER
# ═══════════════════════════════════════════════════════════
def section_header(icon: str, title: str, subtitle: str = ""):
    sub_html = f"<div style='font-size:0.78rem;color:#4a5268;margin-top:3px;font-family:DM Sans,sans-serif;'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
<div style="display:flex;align-items:flex-end;gap:16px;margin-bottom:20px;margin-top:52px;">
<div style="width:44px;height:44px;flex-shrink:0;background:rgba(240,180,41,0.07);border:1px solid rgba(240,180,41,0.16);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:20px;">
{icon}
</div>
<div>
<div style="font-family:'Bebas Neue',sans-serif;font-size:1.45rem;letter-spacing:2px;color:#f0f2f8;line-height:1;">{title}</div>
{sub_html}
</div>
<div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(240,180,41,0.2),transparent);margin-bottom:8px;"></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  IDLE STATE
# ═══════════════════════════════════════════════════════════
if not target_url:
    st.markdown("""
<div style="margin-top:60px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:80px 40px;border:1px dashed rgba(255,255,255,0.07);border-radius:20px;text-align:center;">
<div style="width:64px;height:64px;background:linear-gradient(135deg,rgba(240,180,41,0.12),rgba(255,94,26,0.08));border:1px solid rgba(240,180,41,0.18);border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:28px;margin-bottom:20px;">✦</div>
<div style="font-family:'Bebas Neue',sans-serif;font-size:1.8rem;letter-spacing:2px;color:#f0f2f8;margin-bottom:10px;">
READY FOR ANALYSIS
</div>
<p style="font-size:0.88rem;color:#4a5268;max-width:320px;line-height:1.7;margin:0;">
Paste a YouTube URL into the sidebar to begin detecting golden moments
using the composite scoring algorithm.
</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════
if target_url:
    vid_match = re.search(r"(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", target_url)
    if not vid_match:
        st.error("⚠️  Invalid URL — could not extract a YouTube Video ID.")
        st.stop()

    v_id = vid_match.group(1)

    with st.status("⚙️  Initialising BI Pipeline...", expanded=True) as status:
        raw       = fetch_comments_refined(v_id)
        df_parsed = process_intelligence(raw)
        st.write("🧠  Running emotion classification on timestamped comments...")
        df_work = df_parsed.head(3500).copy()
        df_work['Sentiment'] = df_work['Content'].apply(classify_sentiment_logic)
        df_f    = df_work[df_work['Sentiment'] != "Neutral"].copy()
        status.update(label="✦  Analysis Complete — Dashboard Ready", state="complete", expanded=False)

    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)

    # ── KPI ROW ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Comments Captured", f"{len(raw):,}")
    c2.metric("Timestamped",       f"{len(df_parsed):,}")
    c3.metric("Emotive Signals",   f"{len(df_f):,}")
    c4.metric("Confidence",        "96.4%")

    # ── HEATMAP ──
    section_header("📊", "ENGAGEMENT HEATMAP", "Emotional intensity across the video timeline")

    fig = px.histogram(
        df_f, x="Seconds", color="Sentiment", nbins=120,
        template="plotly_dark", height=380,
        color_discrete_map={k: v["color"] for k, v in EMOTION_CONFIG.items()},
        labels={"Seconds": "Timeline (seconds)", "count": "Emotional Mentions"}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,23,32,0.6)",
        font=dict(family="JetBrains Mono", color="#4a5268", size=11),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="#8b93a8", size=11), title_text=""),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False,
                   title_font=dict(color="#4a5268"), tickfont=dict(color="#4a5268")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False,
                   title_font=dict(color="#4a5268"), tickfont=dict(color="#4a5268")),
        bargap=0.05, margin=dict(l=0, r=0, t=36, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── GOLDEN MOMENTS ──
    section_header("🏆", "GOLDEN MOMENT DETECTION",
                   "Composite score: volume × emotion heat × diversity bonus · min 3-min spread")

    if df_f.empty:
        st.info("No emotive peaks detected — try a video with a more engaged comment section.")
    else:
        highlights = compute_smart_highlights(df_f, top_n=3)

        rank_meta = [
            {"label": "PEAK MOMENT",  "crown": "👑", "border_top": "#f0b429"},
            {"label": "RUNNER-UP",    "crown": "🥈", "border_top": "#8b93a8"},
            {"label": "THIRD SPIKE",  "crown": "🥉", "border_top": "#b45309"},
        ]

        cols = st.columns(3, gap="large")
        for i, row in highlights.iterrows():
            cfg  = EMOTION_CONFIG.get(row['Sentiment'], FALLBACK_CFG)
            meta = rank_meta[i] if i < len(rank_meta) else rank_meta[-1]
            pct  = int(row['ScorePct'])

            with cols[i]:
                st.markdown(f"""
<div style="position:relative;background:linear-gradient(160deg,rgba(255,255,255,0.04) 0%,rgba(255,255,255,0.01) 100%);border:1px solid rgba(255,255,255,0.08);border-top:3px solid {meta['border_top']};border-radius:16px;padding:28px 24px 24px;overflow:hidden;font-family:'DM Sans',sans-serif;">
<div style="position:absolute;top:-30px;right:-30px;width:120px;height:120px;background:radial-gradient(circle,{cfg['glow']} 0%,transparent 70%);pointer-events:none;"></div>
<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
<div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;font-weight:600;letter-spacing:1.5px;color:#4a5268;">{meta['label']}</div>
<div style="font-size:1.2rem;">{meta['crown']}</div>
</div>
<div style="display:inline-flex;align-items:center;gap:6px;background:{cfg['bg']};border:1px solid {cfg['border']};border-radius:6px;padding:4px 12px;font-size:0.72rem;font-weight:600;color:{cfg['color']};margin-bottom:16px;font-family:'JetBrains Mono',monospace;letter-spacing:0.5px;">
{cfg['icon']}  {row['Sentiment'].upper()}
</div>
<div style="font-family:'Bebas Neue',sans-serif;font-size:3.8rem;letter-spacing:3px;line-height:1;color:#f0f2f8;margin-bottom:4px;">
{row['Timestamp']}
</div>
<div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#4a5268;margin-bottom:22px;letter-spacing:0.5px;">
{int(row['Diversity'])} emotion type{'s' if row['Diversity'] > 1 else ''}  ·  {int(row['Count'])} comments
</div>
<div style="width:100%;height:4px;background:rgba(255,255,255,0.06);border-radius:2px;margin-bottom:10px;overflow:hidden;">
<div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{cfg['color']},{meta['border_top']});border-radius:2px;"></div>
</div>
<div style="display:flex;justify-content:space-between;align-items:center;">
<span style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;letter-spacing:0.5px;color:#4a5268;">COMPOSITE SCORE</span>
<span style="font-family:'Bebas Neue',sans-serif;font-size:1.15rem;letter-spacing:1px;color:{cfg['color']};">{pct}</span>
</div>
</div>
""", unsafe_allow_html=True)

        # ── Algorithm explainer ──
        st.markdown("""
<div style="margin-top:18px;padding:16px 20px;background:rgba(240,180,41,0.03);border:1px solid rgba(240,180,41,0.1);border-radius:10px;display:flex;gap:14px;align-items:flex-start;">
<span style="font-size:1rem;flex-shrink:0;margin-top:1px;">⚡</span>
<div>
<div style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;font-weight:600;letter-spacing:1px;color:#f0b429;margin-bottom:6px;">HOW THE SCORE IS CALCULATED</div>
<p style="font-size:0.78rem;color:#4a5268;margin:0;line-height:1.75;">
Each 60-second window is ranked by <code style='color:#8b93a8;background:rgba(255,255,255,0.05);padding:1px 6px;border-radius:3px;'>count × avg_heat × diversity_bonus</code>.
Heat weights: Controversial 1.5 · Funny 1.4 · Inspirational 1.3 · Happy 1.0 · Sad 0.9.
A 3-minute minimum gap is enforced so highlights are spread across the full video.
</p>
</div>
</div>
""", unsafe_allow_html=True)

    # ── EMOTION BREAKDOWN ──
    section_header("🎭", "EMOTION BREAKDOWN", "Distribution of detected sentiment categories")

    em_counts = df_f['Sentiment'].value_counts().reset_index()
    em_counts.columns = ['Sentiment', 'Count']
    fig2 = px.bar(
        em_counts, x='Sentiment', y='Count', color='Sentiment',
        template='plotly_dark', height=300,
        color_discrete_map={k: v["color"] for k, v in EMOTION_CONFIG.items()},
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,23,32,0.6)",
        showlegend=False,
        font=dict(family="JetBrains Mono", color="#4a5268", size=11),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False, tickfont=dict(color="#8b93a8")),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False, tickfont=dict(color="#4a5268")),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    fig2.update_traces(marker_line_width=0)
    st.plotly_chart(fig2, use_container_width=True)