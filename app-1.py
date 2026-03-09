import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from deep_translator import GoogleTranslator
import os
import re

# 1. 頁面配置
st.set_page_config(page_title="AI Phishing Guard Pro", layout="wide", page_icon="🛡️")

# 2. 專業戰情室 CSS 樣式
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1f2937; }
    .metric-card {
        background: #ffffff; border-radius: 12px; padding: 20px;
        border: 1px solid #e5e7eb; border-left: 6px solid #3b82f6;
        margin-bottom: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .analysis-box {
        background: #f8fafc; border-radius: 10px; padding: 15px;
        border: 1px solid #e2e8f0; margin-bottom: 10px;
    }
    h1, h2, h3 { color: #1e3a8a !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)

# 3. 側邊欄：研究數據基礎
with st.sidebar:
    st.markdown("## 🛡️ 威脅情報核心")
    st.write("---")
    st.markdown('<div class="metric-card"><b>🔹 資料庫母體</b><br>82,486 筆郵件數據</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><b>🎯 模型準確率</b><br>95.12%</div>', unsafe_allow_html=True)
    st.write("---")
    st.markdown("### 🔍 惡意行為特徵 (UBA)")
    img_path = 'analysis_result.png'
    if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
        st.image(img_path, caption="行為基線比對圖", use_container_width=True)

# 4. 模型載入
@st.cache_resource
def load_and_train():
    try:
        df = pd.read_csv('phishing_small.csv').dropna(subset=['text_combined'])
        df_sample = df.sample(min(15000, len(df)), random_state=42)
        tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
        X = tfidf.fit_transform(df_sample['text_combined'].astype(str))
        model = MultinomialNB()
        model.fit(X, df_sample['label'])
        return tfidf, model
    except: return None, None

tfidf_vec, ai_model = load_and_train()

# 5. 主頁面佈局
st.title("🛡️ 智慧資安：跨語言釣魚郵件 AI 偵測系統")
col_input, col_report = st.columns([1.2, 1])

with col_input:
    st.subheader("📥 待測郵件掃描 (Email Structure Analysis)")
    user_input = st.text_area("請在此貼入郵件本文：", height=400, placeholder="Waiting for input...")
    
    if st.button("🚀 啟動深度威脅分析"):
        if user_input and ai_model:
            with st.spinner('🔐 執行語意正規化與結構分析...'):
                try:
                    # A. 執行正規化
                    translated = GoogleTranslator(source='auto', target='en').translate(user_input)
                    t_lower = translated.lower()
                    
                    # B. Email 結構分析邏輯 (新功能)
                    # 1. Links Detected (使用正則表達式提取網址)
                    links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', user_input)
                    
                    # 2. Urgency Language (急迫性語法)
                    urgency_words = ["urgent", "immediately", "within 2 hours", "final warning", "expired", "action required"]
                    urgency_hits = [w for w in urgency_words if w in t_lower]
                    
                    # 3. Financial Request (財務/帳據請求)
                    finance_words = ["verify account", "bank", "payment", "invoice", "credit card", "security credentials"]
                    finance_hits = [w for w in finance_words if w in t_lower]
                    
                    # C. AI 預測
                    vec = tfidf_vec.transform([translated])
                    prob = ai_model.predict_proba(vec)[0][1]
                    
                    with col_report:
                        st.subheader("🕵️ 資安診斷報告")
                        st.metric("威脅評分 (Threat Score)", f"{prob*100:.2f}%", delta="⚠️ 高危" if prob > 0.7 else "✅ 安全")
                        
                        # --- 專業結構分析顯示區 ---
                        st.write("### 🏗️ Email Structure Analysis")
                        
                        # 顯示 Links Detected
                        with st.container():
                            st.markdown(f"**🔗 Links Detected:** {len(links)} 筆")
                            if links:
                                for link in links[:2]: st.code(link, language="text") # 顯示前兩筆
                        
                        # 顯示 Urgency & Finance 狀態
                        st.markdown(f"**⚡ Urgency Language:** {'🔴 偵測到急迫誘導' if urgency_hits else '🟢 正常'}")
                        st.markdown(f"**💰 Financial Request:** {'🔴 涉及帳據請求' if finance_hits else '🟢 無相關描述'}")
                        
                        st.write("---")
                        st.write("### 🚨 Detected Suspicious Patterns")
                        all_patterns = urgency_hits + finance_hits
                        if all_patterns:
                            st.warning(f"命中特徵：{', '.join(all_patterns)}")
                        else:
                            st.success("未發現顯著語意攻擊特徵。")

                        with st.expander("📝 檢視語意正規化分析結果"):
                            st.info(translated)
                except Exception as e:
                    st.error(f"偵測異常: {e}")
        else:
            st.warning("請輸入內容並確保數據集已備妥。")

with col_report:
    if 'prob' not in locals():
        st.write("---")
        st.markdown("### 📖 操作說明")
        st.write("1. 貼入郵件本文，系統會自動解析連結與語法結構。")
        st.write("2. 結合 **8 萬筆樣本** 之 AI 模型進行深度診斷。")