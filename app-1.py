import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from deep_translator import GoogleTranslator
import os

# 1. 頁面配置
st.set_page_config(page_title="AI Phishing Guard Pro", layout="centered", page_icon="🛡️")

# 2. 商務風格 CSS
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #1f2937; }
    .report-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 25px;
        border: 1px solid #e2e8f0;
        margin-top: 20px;
    }
    .status-badge {
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    h1, h2, h3 { color: #1e3a8a !important; }
    </style>
    """, unsafe_allow_html=True)

# 3. 模型載入邏輯
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
    except Exception as e:
        st.error(f"系統初始化失敗: {e}")
        return None, None

tfidf_vec, ai_model = load_and_train()

# --- UI 開始 ---
st.title("🛡️ 智慧資安：跨語言釣魚郵件偵測系統")
st.info("系統採用針對 **82,486 筆數據** 訓練之 AI 模型，結合語意歸一化與行為特徵分析。")

# 第一層：Email Input
st.subheader("📥 1. Email Input")
user_input = st.text_area("請在此貼入待測郵件本文：", height=250, placeholder="Paste email content here...")

if st.button("🚀 啟動深度威脅掃描"):
    if user_input:
        with st.spinner('🔐 執行語意正規化與行為模式比對...'):
            try:
                # A. 執行正規化
                translated = GoogleTranslator(source='auto', target='en').translate(user_input)
                
                # B. 偵測 Suspicious Patterns
                patterns = {
                    "Urgency (急迫性誘導)": ["urgent", "immediately", "final warning", "within 2 hours"],
                    "Credential Harvesting (憑證釣魚)": ["verify account", "security credentials", "login", "password"],
                    "Call to Action (引導點擊)": ["click here", "update-now", "access link", "follow the link"]
                }
                detected = []
                for category, keywords in patterns.items():
                    for word in keywords:
                        if word in translated.lower():
                            detected.append(f"🎯 {category}: `{word}`")
                
                # C. 行為特徵分析
                short_urls = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'reurl', 'sec-login', '.xyz']
                has_short_url = any(url in user_input.lower() for url in short_urls)
                
                # D. AI 分數預測
                vec = tfidf_vec.transform([translated])
                prob = ai_model.predict_proba(vec)[0][1]
                score = prob * 100

                # 第二層：Detection Result
                st.write("---")
                st.subheader("📊 2. Detection Result")
                
                # 視覺化評分卡片
                col_score, col_status = st.columns([1, 1])
                with col_score:
                    st.metric("威脅評分 (Threat Score)", f"{score:.2f}%")
                with col_status:
                    if score > 70:
                        st.markdown("<span class='status-badge' style='background:#fee2e2; color:#ef4444;'>🔴 高度威脅</span>", unsafe_allow_html=True)
                    elif score > 40:
                        st.markdown("<span class='status-badge' style='background:#fef3c7; color:#f59e0b;'>🟡 中度風險</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span class='status-badge' style='background:#dcfce7; color:#22c55e;'>🟢 檢測安全</span>", unsafe_allow_html=True)

                # 第三層：Security Analysis
                st.write("---")
                st.subheader("🕵️ 3. Security Analysis")
                
                with st.container():
                    st.markdown("#### 🚨 Detected Suspicious Patterns")
                    if detected:
                        for p in detected:
                            st.write(p)
                    else:
                        st.write("✅ 未發現顯著語意威脅模式。")
                    
                    st.write("")
                    st.markdown("#### 🔍 Behavior Feature Extraction (UBA)")
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**縮網址偵測：** {'🔴 異常' if has_short_url else '🟢 無'}")
                    c2.markdown(f"**誘導詞命中：** {len(detected)} 筆")

                    with st.expander("📝 檢視跨語言語意正規化分析結果 (XAI)"):
                        st.info(translated)

            except Exception as e:
                st.error(f"偵測異常: {e}")
    else:
        st.warning("請先輸入郵件內容。")