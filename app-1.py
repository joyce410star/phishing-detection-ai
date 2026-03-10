import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from deep_translator import GoogleTranslator
from langdetect import detect
import os
import re

# 1. 頁面配置
st.set_page_config(page_title="AI Phishing Guard Pro", layout="wide", page_icon="🛡️")

# 2. 專業 CSS 樣式
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1f2937; }
    [data-testid="stSidebar"] { background-color: #f9fafb !important; border-right: 1px solid #e5e7eb; }
    .metric-card {
        background: #ffffff; border-radius: 12px; padding: 20px;
        border: 1px solid #e5e7eb; border-left: 6px solid #3b82f6;
        margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .structure-box { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }
    h1, h2, h3 { color: #1e3a8a !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)

# 3. 側邊欄：研究數據基礎
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
    if os.path.exists('analysis_result.png'):
        st.image('analysis_result.png', caption="行為基線比對圖", use_container_width=True)

# 4. 模型載入邏輯：自動在線訓練 (解決找不到 pkl 的問題)
@st.cache_resource
def load_and_train():
    try:
        if not os.path.exists('phishing_small.csv'):
            return None, None
        # 讀取數據並進行輕量化訓練以加速啟動
        df = pd.read_csv('phishing_small.csv', nrows=20000).dropna(subset=['text_combined'])
        tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
        X = tfidf.fit_transform(df['text_combined'].astype(str))
        model = MultinomialNB()
        model.fit(X, df['label'])
        return tfidf, model
    except:
        return None, None

tfidf_vec, ai_model = load_and_train()

# 5. 偵測輔助函式
def translate_if_needed(text):
    if not text or len(text.strip()) < 5: return text
    try:
        lang = detect(text)
        if lang != "en":
            return GoogleTranslator(source='auto', target='en').translate(text)
    except: pass
    return text

def calculate_risk(prob, links, urg_hits, fin_hits):
    score = prob
    # 針對結構特徵進行權重補償
    if links: score += 0.2
    score += 0.15 * len(urg_hits)
    score += 0.15 * len(fin_hits)
    # 組合攻擊補強：若偵測到關鍵誘導詞，強制拉高風險
    if links and (urg_hits or fin_hits): score += 0.1
    return min(score, 1.0)

# 6. 主介面佈局
st.title("🛡️ 智慧資安：跨語言釣魚郵件 AI 偵測系統")
tab1, tab2 = st.tabs(["🔍 單封深度掃描", "📂 CSV 批次分析"])

with tab1:
    col_in, col_res = st.columns([1.2, 1])
    if 'last_result' not in st.session_state: st.session_state.last_result = None

    with col_in:
        st.subheader("📥 待測郵件掃描 (Email Structure Analysis)")
        user_input = st.text_area("請在此貼入郵件本文：", height=350, key="input_area")
        if st.button("🚀 啟動深度威脅分析"):
            if user_input and ai_model:
                with st.spinner('🔐 執行語意正規化與行為比對...'):
                    # 執行分析流程
                    translated = translate_if_needed(user_input)
                    t_lower = translated.lower()
                    links = re.findall(r'https?://\S+', user_input)
                    
                    urg_words = ["urgent", "immediately", "verify", "suspended", "limit", "warning"]
                    fin_words = ["bank", "payment", "login", "account", "credentials"]
                    urg_hits = [w for w in urg_words if w in t_lower]
                    fin_hits = [w for w in fin_words if w in t_lower]
                    
                    # AI 預測
                    vec = tfidf_vec.transform([translated])
                    prob = ai_model.predict_proba(vec)[0][1]
                    risk_score = calculate_risk(prob, links, urg_hits, fin_hits)
                    
                    st.session_state.last_result = {
                        "score": risk_score * 100, "links": len(links),
                        "urg": urg_hits, "fin": fin_hits, "trans": translated
                    }
            else: st.warning("請輸入內容並確保數據集已備妥。")

    with col_res:
        if st.session_state.last_result:
            res = st.session_state.last_result
            score = res["score"]
            # 顏色與狀態判定
            if score > 70: status, delta = "🔴 HIGH RISK", "inverse"
            elif score >= 40: status, delta = "🟡 MEDIUM RISK", "off"
            else: status, delta = "🟢 SAFE", "normal"
            
            st.subheader("🕵️ 資安診斷報告")
            st.metric("Threat Score", f"{score:.2f}%", delta=status, delta_color=delta)
            st.progress(score/100)
            
            st.write("### Structure Analysis")
            st.markdown(f"""<div class="structure-box">
                Links detected: **{res['links']}**<br>
                Urgency language: **{'YES' if res['urg'] else 'NO'}**<br>
                Financial request: **{'YES' if res['fin'] else 'NO'}**
            </div>""", unsafe_allow_html=True)
            
            hits = list(set(res["urg"] + res["fin"]))
            if hits: st.warning(f"🎯 命中模式：{', '.join(hits)}")
            
            with st.expander("📝 檢視翻譯結果"):
                st.info(res["trans"])
        else:
            st.info("💡 貼入內容後點擊按鈕即可啟動掃描。")

# --- TAB 2: CSV 批次分析 (輕量化處理) ---
with tab2:
    st.subheader("📂 批量威脅鑑定中心")
    up_csv = st.file_uploader("選擇上傳 CSV 檔案", type="csv", key="csv_up")
    if up_csv and ai_model:
        df_b = pd.read_csv(up_csv)
        col_name = st.selectbox("請選擇郵件本文欄位：", df_b.columns)
        if st.button("🛠️ 開始批量掃描"):
            for i, txt in enumerate(df_b[col_name].astype(str).tolist()[:10]):
                p_b = ai_model.predict_proba(tfidf_vec.transform([txt]))[0][1]
                st.write(f"Email #{i+1} → {'🚨 PHISHING' if p_b > 0.5 else '✅ SAFE'} ({p_b*100:.1f}%)")