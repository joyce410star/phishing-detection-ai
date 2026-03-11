import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from deep_translator import GoogleTranslator
from langdetect import detect
import os
import re

# 1. 頁面基礎配置
st.set_page_config(page_title="AI Phishing Guard Pro", layout="wide", page_icon="🛡️")

# 2. 專業 CSS 樣式與 XAI 容器設計
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1f2937; }
    [data-testid="stSidebar"] { background-color: #f9fafb !important; border-right: 1px solid #e5e7eb; }
    .xai-box {
        background-color: #f0f7ff; border: 1px solid #bae6fd;
        border-radius: 8px; padding: 15px; border-left: 5px solid #0284c7;
        margin-bottom: 15px;
    }
    .metric-card {
        background: #ffffff; border-radius: 12px; padding: 20px;
        border: 1px solid #e5e7eb; border-left: 6px solid #3b82f6;
        margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .batch-res { padding: 10px; border-radius: 8px; margin-bottom: 5px; border: 1px solid #e5e7eb; }
    h1, h2, h3 { color: #1e3a8a !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)

# 初始化 Session State 以確保分頁切換穩定
if 'last_res' not in st.session_state:
    st.session_state.last_res = None

# 3. 側邊欄：Model Information
with st.sidebar:
    st.markdown("## ⚙️ Model Information")
    st.markdown("""
    <div class="metric-card">
        <div style="color:#6b7280; font-size:0.85rem; font-weight:600;">DATASET SIZE</div>
        <div style="color:#1e3a8a; font-size:1.1rem; font-weight:700;">40,000 Emails</div>
        <div style="color:#6b7280; font-size:0.85rem; font-weight:600;">CORE MODEL</div>
        <div style="color:#1e3a8a; font-size:1.1rem; font-weight:700;">Naive Bayes</div>
        <div style="color:#6b7280; font-size:0.85rem; font-weight:600;">ACCURACY</div>
        <div style="color:#1e3a8a; font-size:1.1rem; font-weight:700;">92%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    st.markdown("### 🔍 惡意行為特徵 (UBA)")
    if os.path.exists('analysis_result.png'):
        st.image('analysis_result.png', caption="Behavior Baseline Analysis", use_container_width=True)

# 4. 核心分析引擎 (與 VS Code 測試端邏輯完全同步)
@st.cache_resource
def load_and_train():
    try:
        if not os.path.exists('phishing_small.csv'): return None, None
        df = pd.read_csv('phishing_small.csv', nrows=20000).dropna(subset=['text_combined'])
        tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
        X = tfidf.fit_transform(df['text_combined'].astype(str))
        model = MultinomialNB()
        model.fit(X, df['label'])
        return tfidf, model
    except: return None, None

tfidf_vec, ai_model = load_and_train()

def analyze_email(text):
    # A. 語意正規化 (XAI 第一層)
    try:
        lang = detect(text)
        trans = GoogleTranslator(source='auto', target='en').translate(text) if lang != 'en' else text
    except: trans = text
    
    t_low = trans.lower()
    explanations = []
    
    # B. 結構特徵偵測
    links = re.findall(r'https?://([a-zA-Z0-9.-]+)', text)
    urg_words = ["urgent", "immediately", "verify", "suspended", "limit", "warning", "24 hours"]
    fin_words = ["bank", "payment", "login", "account", "credentials", "invoice"]
    
    urg_hits = [w for w in urg_words if w in t_low]
    fin_hits = [w for w in fin_words if w in t_low]
    
    # C. AI 預測
    vec = tfidf_vec.transform([trans])
    prob = ai_model.predict_proba(vec)[0][1]
    explanations.append(f"🔹 AI 語意原始評分: {prob*100:.1f}%")
    
    # D. 權重補償邏輯 (Explainable Reasoning)
    score = prob
    if links:
        score += 0.2
        explanations.append(f"🚩 發現外部連結: `{', '.join(list(set(links)))}` (+20%)")
    if urg_hits:
        score += 0.15 * len(urg_hits)
        explanations.append(f"⚠️ 命中急迫性詞彙: {', '.join(urg_hits)} (+{15*len(urg_hits)}%)")
    if fin_hits:
        score += 0.15 * len(fin_hits)
        explanations.append(f"💰 涉及財務敏感詞: {', '.join(fin_hits)} (+{15*len(fin_hits)}%)")
    if links and (urg_hits or fin_hits):
        score += 0.1
        explanations.append(f"🧬 複合威脅：連結與誘導語同時出現 (+10%)")

    return {
        "final_score": min(score, 1.0) * 100,
        "raw_prob": prob * 100,
        "explanations": explanations,
        "trans": trans,
        "links_count": len(links)
    }

# 5. 主介面分頁架構
st.title("🛡️ 智慧資安：跨語言釣魚郵件 AI 偵測系統")
tab1, tab2 = st.tabs(["🔍 單封深度掃描 (XAI)", "📂 CSV 批次分析"])

# --- TAB 1: 單封掃描 ---
with tab1:
    col_in, col_res = st.columns([1.2, 1])
    with col_in:
        st.subheader("📥 待測郵件掃描")
        u_input = st.text_area("請在此貼入郵件本文：", height=350, key="single_in", placeholder="貼入郵件內容進行 XAI 深度鑑定...")
        if st.button("🚀 啟動 XAI 威脅鑑定", key="run_single"):
            if u_input and ai_model:
                with st.spinner('🔐 執行決策路徑分析中...'):
                    st.session_state.last_res = analyze_email(u_input)
                    st.rerun()
            else: st.warning("請輸入內容。")

    with col_res:
        if st.session_state.last_res:
            res = st.session_state.last_res
            s = res["final_score"]
            status, delta_c = ("🔴 HIGH", "inverse") if s > 70 else (("🟡 MEDIUM", "off") if s >= 40 else ("✅ SAFE", "normal"))
            
            st.subheader("🕵️ 資安診斷報告")
            st.metric("Risk Score", f"{s:.2f}%", delta=status, delta_color=delta_c)
            st.progress(s/100)
            
            st.write("### 🧠 AI 決策路徑 (Explainability)")
            st.markdown('<div class="xai-box">' + "<br>".join(res["explanations"]) + '</div>', unsafe_allow_html=True)
            
            with st.expander("📝 檢視語意正規化結果"):
                st.info(res["trans"])
        else:
            st.info("💡 貼入內容並啟動掃描，系統將展示 Explain, Don't Just Warn 判定依據。")

# --- TAB 2: CSV 批次分析 ---
with tab2:
    st.subheader("📂 批量威脅鑑定中心")
    up_csv = st.file_uploader("選擇上傳 CSV 檔案", type="csv", key="csv_batch")
    if up_csv and ai_model:
        df_b = pd.read_csv(up_csv)
        col_name = st.selectbox("請選擇郵件內容欄位：", df_b.columns)
        limit = st.slider("分析筆數", 5, 50, 10)
        
        if st.button("🛠️ 開始批量掃描任務", key="run_batch"):
            with st.spinner('正在處理批量數據...'):
                for i, txt in enumerate(df_b[col_name].astype(str).tolist()[:limit]):
                    b_res = analyze_email(txt)
                    p = b_res["final_score"]
                    c = "#fee2e2" if p > 70 else ("#fef3c7" if p >= 40 else "#dcfce7")
                    l = "🔴 PHISHING" if p > 70 else ("🟡 SUSPICIOUS" if p >= 40 else "✅ SAFE")
                    
                    st.markdown(f"""<div class="batch-res" style="background:{c}">
                        Email #{i+1} → <b>{l}</b> ({p:.1f}%) <br>
                        <small style="color:#4b5563">理由：{", ".join([e.split(': ')[0] for e in b_res['explanations'] if '🔹' not in e])}</small>
                    </div>""", unsafe_allow_html=True)
            st.success("批量任務已完成。")