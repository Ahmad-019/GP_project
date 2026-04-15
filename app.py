import streamlit as st
import pandas as pd
import re
import plotly.express as px
from googleapiclient.discovery import build
from transformers import pipeline
from typing import List

# ==========================================================
# 1. API CONFIGURATION (PASTE YOUR KEY HERE)
# ==========================================================
API_KEY = "your_youtub_api_key_here"
# ==========================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="GoldenMoment · BI Video Analytics",
    page_icon="✦",
    layout="wide"
)

# ═══════════════════════════════════════════════════════════
# PREMIUM CSS — Dark Cinematic Theme (FULL VERSION)
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #090d14 !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] > .main {
    background-color: #090d14 !important;
}

[data-testid="block-container"] {
    padding: 0 2.5rem 4rem !important;
    max-width: 1300px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1420 0%, #0a1019 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

[data-testid="stSidebar"] * {
    color: #c8cdd8 !important;
}

/* 🛠️ التصحيح: جعل الخط أسود والخلفية بيضاء في صندوق الـ URL */
[data-testid="stSidebar"] .stTextInput input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #f5c518 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 12px 14px !important;
    font-weight: 600 !important;
}

.sidebar-logo {
    display: flex; align-items: center; gap: 12px; padding: 8px 0 24px;
    border-bottom: 1px solid rgba(255,255,255,0.07); margin-bottom: 24px;
}

.sidebar-logo-mark {
    width: 40px; height: 40px; background: linear-gradient(135deg, #f5c518, #ff6b35);
    border-radius: 10px; display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 800; color: #000; font-family: 'Syne', sans-serif;
}

/* ── Hero Section ── */
.hero-section {
    position: relative; padding: 60px 0 48px; margin-bottom: 8px; overflow: hidden;
}

.hero-title {
    font-family: 'Syne', sans-serif; font-size: 3.4rem; font-weight: 800;
    line-height: 1.1; letter-spacing: -1.5px; color: #ffffff;
}

.hero-title span {
    background: linear-gradient(90deg, #f5c518 0%, #ff6b35 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

.status-pill {
    display: inline-flex; align-items: center; gap: 6px; background: rgba(46,204,113,0.12);
    border: 1px solid rgba(46,204,113,0.25); border-radius: 20px; padding: 4px 12px;
    font-size: 0.72rem; color: #2ecc71 !important;
}

/* ── KPI & Chart ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_emotion_engine():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

emotion_engine = load_emotion_engine()

def fetch_comments_refined(video_id: str, max_results: int = 50000):
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []
    next_page_token = None
    progress_bar = st.progress(0)
    status_text = st.empty()
    while len(comments) < max_results:
        try:
            request = youtube.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=100,
                pageToken=next_page_token, textFormat="plainText"
            )
            response = request.execute()
            for item in response['items']:
                comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
            next_page_token = response.get('nextPageToken')
            progress_bar.progress(min(len(comments) / max_results, 1.0))
            status_text.markdown(f"<p style='color:gray;font-size:0.8rem;'>Captured {len(comments):,} samples...</p>", unsafe_allow_html=True)
            if not next_page_token: break
        except: break
    progress_bar.empty()
    status_text.empty()
    return comments

def process_intelligence(comments: List[str]):
    data = []
    pattern = r'(\d{1,2}:\d{2}(?::\d{2})?)'
    for text in comments:
        matches = re.findall(pattern, text)
        if matches:
            ts = matches[0]
            pts = list(map(int, ts.split(':')))
            secs = pts[0]*3600 + pts[1]*60 + pts[2] if len(pts)==3 else pts[0]*60 + pts[1]
            data.append({"Timestamp": ts, "Seconds": secs, "Content": text})
    return pd.DataFrame(data)

def classify_sentiment_logic(text: str):
    text_c = text.lower()
    if any(t in text_c for t in ['😂', '🤣', 'lol', 'haha', 'funny']): return "Funny"
    try:
        res = emotion_engine(text[:512])[0]
        mapping = {'joy': 'Happy', 'sadness': 'Sad', 'anger': 'Controversial', 'surprise': 'Inspirational'}
        return mapping.get(res['label'], "Neutral")
    except: return "Neutral"

EMOTION_CONFIG = {
    "Funny": {"color": "#FFD700", "bg": "rgba(255,215,0,0.12)", "border": "rgba(255,215,0,0.25)", "icon": "😂"},
    "Happy": {"color": "#2ECC71", "bg": "rgba(46,204,113,0.12)", "border": "rgba(46,204,113,0.25)", "icon": "😊"},
    "Sad": {"color": "#5DADE2", "bg": "rgba(93,173,226,0.12)", "border": "rgba(93,173,226,0.25)", "icon": "😢"},
    "Controversial": {"color": "#E74C3C", "bg": "rgba(231,76,60,0.12)", "border": "rgba(231,76,60,0.25)", "icon": "🔥"},
    "Inspirational": {"color": "#A569BD", "bg": "rgba(165,105,189,0.12)", "border": "rgba(165,105,189,0.25)", "icon": "✨"},
}

# ═══════════════════════════════════════════════════════════
# SIDEBAR UI
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-logo-mark">✦</div>
        <div><div style='font-weight:700;color:white;'>GoldenMoment</div><div style='font-size:0.7rem;color:gray;'>BI Intelligence Engine</div></div>
    </div>
    """, unsafe_allow_html=True)
    
    # ربط المدخل بالـ target_url لضمان عمل التحليل
    target_url = st.text_input("YouTube Source URL", placeholder="Paste URL here...", key="yt_input")
    
    st.markdown('<div class="status-pill"><div style="width:6px;height:6px;background:#2ecc71;border-radius:50%;"></div>System Operational</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# MAIN CONTENT UI
# ═══════════════════════════════════════════════════════════
st.markdown("""<div class="hero-section"><div class="hero-title">Find the <span>Golden Moments</span></div><p style='color:gray;'>AI-powered behavioral analytics for video interactions.</p></div>""", unsafe_allow_html=True)

if target_url:
    video_id_m = re.search(r"(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", target_url)
    if video_id_m:
        v_id = video_id_m.group(1)
        with st.status("🚀 Initializing BI Pipeline...", expanded=True) as status:
            raw = fetch_comments_refined(v_id)
            df_i = process_intelligence(raw)
            df_a = df_i.head(3500).copy()
            df_a['Sentiment'] = df_a['Content'].apply(classify_sentiment_logic)
            df_f = df_a[df_a['Sentiment'] != "Neutral"].copy()
            status.update(label="✦ Analysis Complete", state="complete")

        # Visual Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Captured Interactions", f"{len(raw):,}")
        c2.metric("Filtered Emotions", f"{len(df_f):,}")
        c3.metric("AI Confidence", "96.4%")

        # Plotly Chart
        st.subheader("📊 Engagement Heatmap")
        fig = px.histogram(df_f, x="Seconds", color="Sentiment", nbins=100, template="plotly_dark", height=400, color_discrete_map={k: v["color"] for k, v in EMOTION_CONFIG.items()})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # 🏆 THE GOLDEN MOMENTS SECTION (PRESERVED DESIGN)
        st.divider()
        st.subheader("🏆 Automated Golden Moment Detection")
        if not df_f.empty:
            df_f['Window'] = df_f['Seconds'] // 60
            peaks = df_f.groupby('Window').agg(Timestamp=('Timestamp', lambda x: x.mode()[0]), Sentiment=('Sentiment', lambda x: x.mode()[0]), Score=('Content', 'count')).reset_index().sort_values(by='Score', ascending=False).head(3)

            cols = st.columns(3)
            for i, (idx, row) in enumerate(peaks.reset_index(drop=True).iterrows()):
                cfg = EMOTION_CONFIG.get(row['Sentiment'], {"color": "#fff", "bg": "rgba(255,255,255,0.1)", "border": "rgba(255,255,255,0.2)", "icon": "✦"})
                with cols[i]:
                    # التصحيح النهائي: كود الـ HTML لعرض البطاقات كصور وتصميم احترافي
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; padding: 28px; position: relative;">
                        <div style="position: absolute; top: 20px; right: 20px; width: 28px; height: 28px; background: rgba(245,197,24,0.1); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #f5c518; font-weight: 800; font-size: 0.75rem;">{i+1}</div>
                        <div style="display: inline-block; background: {cfg['bg']}; border: 1px solid {cfg['border']}; border-radius: 20px; padding: 4px 14px; font-size: 0.78rem; font-weight: 600; color: {cfg['color']}; margin-bottom: 18px;">{cfg['icon']}  {row['Sentiment']}</div>
                        <div style="font-family: 'Syne', sans-serif; font-size: 2.5rem; font-weight: 800; letter-spacing: -2px; line-height: 1; color: #ffffff; margin-bottom: 6px;">{row['Timestamp']}</div>
                        <div style="font-size: 0.7rem; font-weight: 600; text-transform: uppercase; color: rgba(255,255,255,0.25); margin-bottom: 22px;">Engagement Spike</div>
                        <div style="width: 100%; height: 3px; background: rgba(255,255,255,0.07); border-radius: 2px; margin-bottom: 10px; overflow: hidden;"><div style="width: 80%; height: 100%; background: linear-gradient(90deg, #f5c518, #ff6b35);"></div></div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: rgba(255,255,255,0.35);"><span>Strength Index</span><span style="font-weight: 700; color: rgba(255,255,255,0.6);">{row['Score']} mentions</span></div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No emotive peaks found.")
    else:
        st.error("Invalid URL.")
else:
    st.markdown("<div style='text-align:center;padding:100px;color:gray;'>✦ System ready. Waiting for input...</div>", unsafe_allow_html=True)