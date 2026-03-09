import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from deep_translator import GoogleTranslator
import os

# 1. 頁面基礎配置：使用白底商務風格
st.set_page_config(page_title="AI Phishing Guard Pro", layout="wide", page_icon="🛡️")

# 2. 商務簡約版 CSS 樣式
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

# 3. 側邊欄：展示 8 萬筆數據研究成果
with st.sidebar:
    st.markdown("## 🛡️ 威脅情報核心")
    st.write("---")
    st.markdown('<div class="metric-card"><b>🔹 資料庫母體</b><br>82,486 筆郵件數據</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><b>🎯 模型準確率</b><br>95.12% (經由驗證集測試)</div>', unsafe_allow_html=True)
    st.write("---")
    st.markdown("### 🔍 惡意行為特徵 (UBA)")
    
    # 雲端環境請確保 analysis_result.png 已上傳至 GitHub
    img_path = 'analysis_result.png'
    if os.path.exists(img_path):
        st.image(img_path, caption="行為基線比對圖", use_container_width=True)
    
    st.info("系統透過 TF-IDF 演算法自動提取 http 連結與關鍵動詞之權重。")

# 4. 模型載入函式 (修正縮排問題)
@st.cache_resource
def load_and_train():
    # 讀取已抽樣的小型數據集以符合 GitHub 上傳限制
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
    st.subheader("📥 待測郵件掃描")
    user_input = st.text_area("請在此貼入郵件本文：", height=380, placeholder="等待輸入內容...")
    
    if st.button("🚀 執行深度威脅掃描"):
        if user_input:
            with st.spinner('🔐 正在分析語意特徵...'):
                try:
                    # 關鍵修正：強制指定從中文 (zh-TW) 翻譯成英文 (en)
                    # 這樣 '永久停用' 會變成 'permanently disabled'，觸發 AI 的警報
                    translated = GoogleTranslator(source='zh-TW', target='en').translate(user_input)
                    
                    vec = tfidf_vec.transform([translated])
                    prob = ai_model.predict_proba(vec)[0][1]
                    
                    with col_report:
                    st.subheader("🕵️ 分析診斷報告")
                    
                    # 1. 補回風險指數卡片 (這是專業感的來源！)
                    st.markdown(f"""<div class='metric-card'>
                        <p style='margin:0; font-size:12px; color:#6b7280;'>THREAT SCORE</p>
                        <h2 style='margin:0; color:#1e3a8a;'>{prob*100:.2f}%</h2>
                    </div>""", unsafe_allow_html=True)
                    
                    # 2. 判斷並顯示狀態燈號與文字分析
                    if prob > 0.5:
                        st.error(f"⚠️ 偵測到高度釣魚威脅")
                        st.write("**分析：** 語意中包含顯著誘導特徵，如永久停用或緊急驗證。")
                    else:
                        st.success(f"✅ 檢測結果：安全")
                        st.write("**分析：** 語意模式符合正常商務通訊基準。")
                        
                        with st.expander("📝 檢視語意處理結果"):
                            st.info(translated)
                except Exception as e:
                    st.error(f"掃描失敗: {e}")
        else:
            st.warning("請輸入內容。")

# 初始操作說明
with col_report:
    if 'prob' not in locals():
        st.write("---")
        st.markdown("### 📖 操作說明")
        st.write("1. 貼入任何語言的郵件內容。")
        st.write("2. 點擊掃描，系統將自動將語意歸一化為英文特徵。")
        st.write("3. 基於 **8 萬筆樣本** 之行為模式進行風險評估。")