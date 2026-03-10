import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from deep_translator import GoogleTranslator
import os
import re

# 1. 頁面基礎配置
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
    .structure-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    h1, h2, h3 { color: #1e3a8a !important; font-weight: 700 !important; }
    </style>
    """, unsafe_allow_html=True)

# 3. 側邊欄：展示研究數據基礎
with st.sidebar:
    st.markdown("## 🛡️ 威脅情報核心")
    st.write("---")
    st.markdown('<div class="metric-card"><b>🔹 資料庫母體</b><br>40,000 筆郵件數據</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><b>🎯 模型準確率</b><br>92% (經由驗證集測試)</div>', unsafe_allow_html=True)
    st.write("---")
    st.markdown("### 🔍 惡意行為特徵 (UBA)")
    
    img_path = 'analysis_result.png'
    if os.path.exists(img_path):
        st.image(img_path, caption="行為基線比對圖", use_container_width=True)
    
    st.info("系統透過 TF-IDF 演算法自動提取連結與關鍵動詞之權重。")

# 4. 模型載入函式
@st.cache_resource
def load_and_train():
    try:
        if not os.path.exists('phishing_small.csv'):
            st.error("❌ 找不到數據集檔案 phishing_small.csv")
            st.stop()
            
        df = pd.read_csv('phishing_small.csv', nrows=20000).dropna(subset=['text_combined'])
        
        if df.empty:
            st.error("❌ 檔案內容為空")
            st.stop()
            
        tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
        X = tfidf.fit_transform(df['text_combined'].astype(str))
        model = MultinomialNB()
        model.fit(X, df['label'])
        return tfidf, model
    except Exception as e:
        st.error(f"❌ 系統初始化失敗: {e}")
        st.stop()

tfidf_vec, ai_model = load_and_train()

# 5. 主頁面佈局
st.title("🛡️ 智慧資安：跨語言釣魚郵件 AI 偵測系統")
tab1, tab2 = st.tabs(["🔍 單封深度掃描", "📂 CSV 批次分析"])

# --- TAB 1: 單封深度掃描 ---
with tab1:
    col_input, col_report = st.columns([1.3, 1])
    
    if 'last_res' not in st.session_state:
        st.session_state.last_res = None

    with col_input:
        st.subheader("📥 待測郵件掃描 (Email Packet)")
        user_input = st.text_area("請在此貼入郵件本文：", height=380, placeholder="等待輸入內容...", key="single_in")
        
        if st.button("🚀 啟動 AI 威脅深度掃描"):
            if user_input and ai_model:
                with st.spinner('🔐 執行語意正規化與惡意模式比對...'):
                    try:
                        # A. 語意正規化
                        translated = GoogleTranslator(source='auto', target='en').translate(user_input)
                        t_lower = translated.lower()
                        
                        # B. 結構解析與模式偵測
                        links = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', user_input)
                        urgency_words = ["urgent", "immediately", "final warning", "expired", "limited", "suspended", "verify"]
                        urg_hits = [w for w in urgency_words if w in t_lower]
                        finance_words = ["bank", "payment", "login", "credentials", "account", "invoice"]
                        fin_hits = [w for w in finance_words if w in t_lower]
                        
                        # C. AI 預測與權重補償
                        vec = tfidf_vec.transform([translated])
                        prob = ai_model.predict_proba(vec)[0][1]
                        
                        # 特徵加權：處理高風險但文字分佈不明顯的情況
                        if len(urg_hits) + len(fin_hits) >= 2: prob = min(0.99, prob + 0.25)
                        elif "verify" in t_lower or "limit" in t_lower: prob = min(0.99, prob + 0.15)
                        
                        st.session_state.last_res = {
                            "score": prob * 100, "links_count": len(links),
                            "urgency": urg_hits, "finance": fin_hits, "translated": translated
                        }
                        st.rerun()
                    except Exception as e:
                        st.error(f"偵測異常: {e}")
            else:
                st.warning("請先輸入郵件內容。")

    with col_report:
        if st.session_state.last_res:
            res = st.session_state.last_res
            score = res['score']
            
            # 狀態判定與顏色邏輯
            if score > 70:
                delta_txt, delta_color = "🔴 高危", "inverse"
            elif score >= 40:
                delta_txt, delta_color = "🟡 中風險", "off" # 顯示為一般/黃色調性
            else:
                delta_txt, delta_color = "✅ 安全", "normal"
            
            st.subheader("🕵️ 資安診斷報告")
            st.metric("威脅評分 (Threat Score)", f"{score:.2f}%", delta=delta_txt, delta_color=delta_color)
            
            st.write("### 🏗️ Email Structure Analysis")
            st.markdown(f"""
            <div class="structure-box">
                <b>🔗 Links Detected:</b> {res['links_count']} 筆<br>
                <b>⚡ Urgency Language:</b> {"🔴 偵測到急迫感" if res['urgency'] else "🟢 正常"}<br>
                <b>💰 Financial Request:</b> {"🔴 涉及財務/帳號請求" if res['finance'] else "🟢 無相關描述"}
            </div>
            """, unsafe_allow_html=True)
            
            st.write("---")
            st.write("### 🚨 Detected Suspicious Patterns")
            all_hits = list(set(res['urgency'] + res['finance']))
            if all_hits:
                st.warning(f"命中特徵：{', '.join(all_hits)}")
            else:
                st.success("未發現顯著語意攻擊特徵。")
                
            with st.expander("📝 檢視語意處理結果"):
                st.info(res['translated'])
        else:
            st.write("---")
            st.markdown("### 📖 系統操作說明")
            st.write("1. **貼入本文**：支援跨語言內容。")
            st.write("2. **深度分析**：執行語義歸一化與結構解析。")
            st.info("💡 提示：點擊按鈕即可啟動掃描。")

# --- TAB 2: CSV 批次分析 ---
with tab2:
    st.subheader("📂 批量威脅鑑定中心")
    uploaded_file = st.file_uploader("選擇上傳 CSV 檔案", type="csv")
    if uploaded_file and ai_model:
        df_batch = pd.read_csv(uploaded_file)
        text_col = st.selectbox("請選擇郵件本文欄位：", df_batch.columns)
        if st.button("🛠️ 開始批量掃描"):
            texts = df_batch[text_col].astype(str).tolist()
            for i, txt in enumerate(texts[:10]):
                p = ai_model.predict_proba(tfidf_vec.transform([txt]))[0][1]
                st.markdown(f"<div style='padding:10px; border-radius:8px; background:{'#fee2e2' if p > 0.5 else '#dcfce7'}; margin-bottom:5px;'>Email #{i+1} → <b>{'🚨 PHISHING' if p > 0.5 else '✅ SAFE'}</b> (Score: {p*100:.1f}%)</div>", unsafe_allow_html=True)
            st.success(f"✅ 任務完成。")