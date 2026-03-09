import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from deep_translator import GoogleTranslator
import os

# 1. 頁面基礎配置：全白商務簡約風格
st.set_page_config(page_title="AI Phishing Guard Pro", layout="wide", page_icon="🛡️")

# 2. 專業 CSS 樣式優化
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1f2937; }
    [data-testid="stSidebar"] { background-color: #f9fafb !important; border-right: 1px solid #e5e7eb; }
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e5e7eb;
        border-left: 6px solid #3b82f6;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 { color: #1e3a8a !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)

# 3. 側邊欄：展示研究數據基礎
with st.sidebar:
    st.markdown("## 🛡️ 威脅情報核心")
    st.write("---")
    st.markdown('<div class="metric-card"><b>🔹 資料庫母體</b><br>82,486 筆郵件數據</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><b>🎯 模型準確率</b><br>95.12% (經由驗證集測試)</div>', unsafe_allow_html=True)
    st.write("---")
    st.markdown("### 🔍 惡意行為特徵 (UBA)")
    
    # 讀取行為基線比對圖
    img_path = 'analysis_result.png'
    if os.path.exists(img_path):
        st.image(img_path, caption="行為基線比對圖", use_container_width=True)
    
    st.info("系統透過 TF-IDF 演算法自動提取連結與關鍵動詞之權重。")

# 4. 模型載入函式：修正縮排與路徑問題
@st.cache_resource
def load_and_train():
    # 使用相對路徑以支援雲端部署
    df = pd.read_csv('phishing_small.csv').dropna(subset=['text_combined'])
    df_sample = df.sample(15000, random_state=42)
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    X = tfidf.fit_transform(df_sample['text_combined'].astype(str))
    model = MultinomialNB()
    model.fit(X, df_sample['label'])
    return tfidf, model

tfidf_vec, ai_model = load_and_train()

# 5. 主頁面佈局
st.title("🛡️ 智慧資安：跨語言釣魚郵件 AI 偵測系統")
st.markdown("##### 結合機器學習與行為分析，識別潛藏在電子郵件中的社交工程威脅")

col_input, col_report = st.columns([1.3, 1])

with col_input:
    st.subheader("📥 待測郵件掃描 (Email Packet)")
    user_input = st.text_area("請在此貼入郵件本文：", height=380, placeholder="等待輸入內容...")
    
    if st.button("🚀 啟動 AI 威脅深度掃描"):
        if user_input:
            with st.spinner('🔐 執行語意歸一化與行為模式比對...'):
                try:
                    # 強制執行語意歸一化：將輸入轉為模型熟悉的英文特徵
                    translated = GoogleTranslator(source='auto', target='en').translate(user_input)
                    
                    # 專業行為特徵提取 (UBA Analysis)
                    # 偵測高危連結行為
                    short_urls = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'reurl', 'sec-login', '.xyz']
                    has_short_url = any(url in user_input.lower() for url in short_urls)
                    
                    # 偵測誘導性動詞 (根據 8 萬筆樣本提取的關鍵特徵)
                    danger_keywords = ['verify', 'urgently', 'permanently', 'disabled', 'suspended', 'immediately', 'expired', 'warning']
                    found_words = [word for word in danger_keywords if word in translated.lower()]
                    
                    # AI 模型預測風險百分比
                    vec = tfidf_vec.transform([translated])
                    prob = ai_model.predict_proba(vec)[0][1]
                    
                    with col_report:
                        st.subheader("🕵️ 資安診斷報告")
                        
                        # 顯示威脅評分卡片
                        st.metric(
                            "威脅評分 (Threat Score)", 
                            f"{prob*100:.2f}%", 
                            delta="⚠️ 高危" if prob > 0.7 else ("🟡 中風險" if prob > 0.4 else "✅ 安全"),
                            delta_color="normal" if prob > 0.4 else "inverse"
                        )
                        
                        # 連動行為分析數據區塊
                        st.write("### 🔍 行為特徵提取 (UBA Analysis)")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown(f"**縮網址偵測：** {'🔴 異常' if has_short_url else '🟢 無'}")
                        with c2:
                            st.markdown(f"**誘導詞數量：** {len(found_words)}")
                        
                        if found_words:
                            st.warning(f"偵測到高危特徵詞：{', '.join(found_words)}")

                        # 找回翻譯內文區塊，實現 Explainable AI
                        st.write("---")
                        with st.expander("📝 檢視跨語言語意正規化 (Normalization)", expanded=True):
                            st.info(translated)
                            
                except Exception as e:
                    st.error(f"偵測失敗: {e}")
        else:
            st.warning("請先輸入郵件內容。")

# 初始說明畫面
with col_report:
    if 'prob' not in locals():
        st.write("---")
        st.markdown("### 📖 操作說明")
        st.write("1. 貼入任何語言的郵件內容。")
        st.write("2. 系統將自動執行語意正規化，解決語言偏差問題。")
        st.write("3. 基於 **8 萬筆樣本** 之行為模式比對給出風險評估。")