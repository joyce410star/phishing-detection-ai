import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from deep_translator import GoogleTranslator
from langdetect import detect
import os
import re

# 1. 頁面配置
st.set_page_config(page_title="AI Scam Guard Pro", layout="wide", page_icon="🛡️")

# 2. 專業 CSS (新增平台標籤樣式)
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1f2937; }
    .xai-box { background-color: #f0f7ff; border-radius: 8px; padding: 15px; border-left: 5px solid #0284c7; margin-bottom: 15px; }
    .platform-tag { padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; color: white; }
    .metric-card { background: #ffffff; border-radius: 12px; padding: 20px; border: 1px solid #e5e7eb; border-left: 6px solid #3b82f6; margin-bottom: 15px; }
    </style>
    """, unsafe_allow_html=True)

if 'last_res' not in st.session_state:
    st.session_state.last_res = None

# 3. 核心分析引擎：新增平台分析邏輯
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

def analyze_scam(text, platform):
    # A. 語意正規化
    try:
        lang = detect(text)
        trans = GoogleTranslator(source='auto', target='en').translate(text) if lang != 'en' else text
    except: trans = text
    
    t_low = trans.lower()
    explanations = []
    
    # B. 平台特徵關鍵字庫
    patterns = {
        "LINE/社交": ["investment", "teacher", "profit", "group", "earn money", "crypto"],
        "SMS/簡訊": ["package", "delivery", "failed", "unpaid", "click to view", "verification code"],
        "通用釣魚": ["urgent", "verify", "suspended", "account", "login", "credentials"]
    }
    
    hits = [w for w in (patterns["LINE/社交"] + patterns["SMS/簡訊"] + patterns["通用釣魚"]) if w in t_low]
    links = re.findall(r'https?://([a-zA-Z0-9.-]+)', text)
    
    # C. AI 預測與權重補償
    prob = ai_model.predict_proba(tfidf_vec.transform([trans]))[0][1]
    score = prob
    
    # D. 跨平台加權邏輯 (重點：針對不同來源調整權重)
    if platform == "LINE / 社群":
        if any(w in t_low for w in ["investment", "profit", "teacher"]):
            score += 0.3
            explanations.append("📈 **偵測到社交平台典型投資詐騙語法 (+30%)**")
    elif platform == "SMS / 簡訊":
        if any(w in t_low for w in ["package", "delivery", "unpaid"]):
            score += 0.25
            explanations.append("📦 **偵測到簡訊典型包裹/欠費詐騙特徵 (+25%)**")
    
    if links:
        score += 0.2
        explanations.append(f"🔗 **包含可疑連結指向:** `{links[0]}` (+20%)")
    
    # 判定詐騙類型
    scam_type = "一般釣魚"
    if any(w in t_low for w in ["investment", "profit"]): scam_type = "投資詐騙"
    elif any(w in t_low for w in ["package", "delivery"]): scam_type = "包裹/代收詐騙"
    elif any(w in t_low for w in ["login", "verify"]): scam_type = "帳據竊取"

    return {
        "final_score": min(score, 1.0) * 100,
        "type": scam_type,
        "explanations": explanations,
        "hits": hits,
        "trans": trans
    }

# 4. 主介面
st.title("🛡️ 智慧資安：全通路詐騙 AI 偵測系統")
st.markdown("##### 支援 Email、LINE、SMS 簡訊與社交媒體內容鑑定")

tab1, tab2 = st.tabs(["🔍 單條訊息鑑定", "📂 CSV 批次分析"])

# --- TAB 1: 單條訊息鑑定 ---
with tab1:
    col_in, col_res = st.columns([1.2, 1])
    
    with col_in:
        st.subheader("📥 訊息內容輸入")
        
        # 1. 取得使用者選擇的平台
        platform = st.selectbox("請選擇訊息來源：", ["Email", "LINE / 社群", "SMS / 簡訊"])
        
        # 2. 根據平台設定動態提示詞 (Placeholder)
        placeholders = {
            "Email": "請在此貼入電子郵件本文，例如：您的帳戶存取權限已被暫時限制...",
            "LINE / 社群": "例如：我是林老師，這是一個穩賺不賠的投資機會，請加入群組...",
            "SMS / 簡訊": "例如：您有一件包裹未領取，請點擊連結查看詳情：http://bit.ly/fake..."
        }
        
        # 3. 將動態提示詞帶入 text_area
        u_input = st.text_area(
            "請在此貼入訊息本文：", 
            height=250, 
            placeholder=placeholders[platform], # 這裡會根據選擇自動切換
            key="single_input"
        )
        
        if st.button("🚀 啟動跨平台 AI 鑑定"):
            # ... 後續分析邏輯保持不變 ...
            if u_input and ai_model:
                with st.spinner('🔐 正在分析決策路徑...'):
                    st.session_state.last_res = analyze_scam(u_input, platform)
                    st.rerun()
            else: st.warning("請輸入內容。")

    with col_res:
        if st.session_state.last_res:
            res = st.session_state.last_res
            s = res["final_score"]
            status = "🔴 HIGH" if s > 70 else ("🟡 MED" if s >= 40 else "✅ SAFE")
            
            st.subheader("🕵️ 鑑定報告")
            st.metric("Scam Probability", f"{s:.2f}%", delta=status)
            
            # 顯示類型標籤
            color = "#ef4444" if s > 70 else "#f59e0b"
            st.markdown(f"**判定類型：** <span class='platform-tag' style='background:{color}'>{res['type']}</span>", unsafe_allow_html=True)
            
            st.write("### 🧠 AI 可解釋性分析 (XAI)")
            st.markdown('<div class="xai-box">' + ("<br>".join(res["explanations"]) if res["explanations"] else "🟢 未發現顯著人為權重補償，純模型判定。") + '</div>', unsafe_allow_html=True)
            
            if res["hits"]:
                st.warning(f"🎯 **關鍵詞命中：** {', '.join(res['hits'])}")
        else:
            st.info("💡 選擇平台並輸入訊息後，AI 將拆解決策原因。")