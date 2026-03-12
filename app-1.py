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
    # --- A. 語意正規化 (你原本的內容) ---
    try:
        lang = detect(text)
        trans = GoogleTranslator(source='auto', target='en').translate(text) if lang != 'en' else text
    except: trans = text
    t_low = trans.lower()
    
    # --- B. 定義判斷因子與權重 (你原本的內容) ---
    reasons = []
    p_weights = {
        "LINE / 社群": ["investment", "profit", "teacher", "group", "earn money", "飆股", "獲利"],
        "SMS / 簡訊": ["package", "delivery", "failed", "unpaid", "verification", "領取", "未繳"],
        "Email": ["urgent", "verify", "suspended", "account", "login", "credentials", "限制"]
    }
    hits = [w for w in p_weights[platform] if w in t_low or w in text]
    links = re.findall(r'https?://([a-zA-Z0-9.-]+)', text)
    prob = ai_model.predict_proba(tfidf_vec.transform([trans]))[0][1]
    
    # --- C. 決策路徑追蹤 (你原本的內容) ---
    final_score = prob
    if hits:
        weight = 0.15 * len(set(hits))
        final_score += weight
        reasons.append(f"🎯 出現高風險關鍵詞：{', '.join(list(set(hits)))}")
    if links:
        final_score += 0.2
        reasons.append(f"🔗 包含可疑外部連結：`{links[0]}`")
    if any(w in t_low for w in ["urgent", "immediately", "24 hours", "立即", "趕快"]):
        final_score += 0.1
        reasons.append("⏳ 要求緊急行動 (Urgency detected)")

    # --- 🌟 重點：新增類型判定 (解決 KeyError) ---
    scam_type = "一般威脅" 
    if any(w in t_low for w in ["investment", "profit", "飆股", "投資"]):
        scam_type = "投資詐騙"
    elif any(w in t_low for w in ["package", "delivery", "包裹", "領取"]):
        scam_type = "包裹/代收詐騙"
    elif any(w in t_low for w in ["login", "verify", "password", "驗證"]):
        scam_type = "帳據竊取"

    # --- 🌟 重點：回傳時補上 "type" ---
    # --- 在 analyze_scam 函式的最後面 ---
    return {
        "final_score": min(final_score, 1.0) * 100,
        "explanations": reasons, # 把原本的 reasons 改名為 explanations
        "trans": trans,
        "type": scam_type
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
            
            st.subheader("🕵️ 鑑定報告")
            st.metric("Scam Probability", f"{s:.2f}%")
            
            # --- 新增：判斷原因區塊 ---
            st.write("### 📝 判斷原因 (Explainable AI)")
            if res["reasons"]:
                for reason in res["reasons"]:
                    st.markdown(f"* {reason}")
            else:
                st.write("🟢 AI 基於整體語意判定為安全，未命中特定惡意特徵。")
            # ----------------------------------------------
            
            with st.expander("📝 檢視語意處理結果"):
                st.info(res["trans"])
            
            # 顯示類型標籤
            color = "#ef4444" if s > 70 else "#f59e0b"
            st.markdown(f"**判定類型：** <span class='platform-tag' style='background:{color}'>{res['type']}</span>", unsafe_allow_html=True)
            
            st.write("### 🧠 AI 可解釋性分析 (XAI)")
            st.markdown('<div class="xai-box">' + ("<br>".join(res["explanations"]) if res["explanations"] else "🟢 未發現顯著人為權重補償，純模型判定。") + '</div>', unsafe_allow_html=True)
            
            if res["hits"]:
                st.warning(f"🎯 **關鍵詞命中：** {', '.join(res['hits'])}")
        else:
            st.info("💡 選擇平台並輸入訊息後，AI 將拆解決策原因。")
with tab2:
    st.subheader("📂 批量威脅鑑定中心")
    # 給予獨立 key，確保分頁切換時組件不會消失
    up_csv = st.file_uploader("選擇上傳 CSV 檔案", type="csv", key="csv_file_up")
    
    if up_csv:
        df_b = pd.read_csv(up_csv)
        # 讓使用者選擇批次數據的來源平台
        batch_plat = st.selectbox("這批數據的來源類型：", ["Email", "LINE / 社群", "SMS / 簡訊"], key="b_plat")
        col_name = st.selectbox("選擇內容欄位：", df_b.columns, key="b_col")
        
        if st.button("🛠️ 開始批量掃描任務", key="b_run"):
            if ai_model:
                for i, txt in enumerate(df_b[col_name].astype(str).tolist()[:10]):
                    res_b = analyze_scam(txt, batch_plat)
                    p = res_b["final_score"]
                    # 根據風險顯示顏色背景
                    c = "#fee2e2" if p > 70 else ("#fef3c7" if p >= 40 else "#dcfce7")
                    st.markdown(f'<div class="batch-res" style="background:{c}; padding:10px; border-radius:8px; margin-bottom:5px;">Email #{i+1} → <b>{p:.1f}%</b></div>', unsafe_allow_html=True)
            else:
                st.error("模型尚未準備就緒，請檢查數據集。")