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

P_WEIGHTS = {
    "LINE / 社群": ["investment", "profit", "teacher", "group", "飆股", "獲利", "加LINE", "領取"],
    "SMS / 簡訊": ["package", "delivery", "unpaid", "verification", "領取", "未繳", "罰鍰", "更新資料"],
    "Email": ["invoice", "overdue", "payment", "suspended", "verify", "security", "unusual", "login", "activity", "identity","發票", "欠費", "逾期", "限制"]
}

def analyze_scam(text, platform):
    # 🌟 1. 強制語意正規化 (確保翻譯)
    try:
        # 強制使用 Google Translator 轉成英文
        # 我們不再比對 text，而是直接使用 trans
        trans = GoogleTranslator(source='auto', target='en').translate(text)
        
        # 建立顯示內容：標註偵測到的狀態
        # 如果翻譯後的內容跟原文不一樣，代表翻譯成功
        if trans.strip().lower() != text.strip().lower():
            display_text = f"【語意分析：偵測為非英語內容，已完成正規化翻譯】\n\n{trans}"
        else:
            display_text = f"【語意分析：偵測為英語內容，維持原始文本分析】\n\n{text}"
            
    except Exception as e:
        # 如果翻譯發生錯誤 (例如網路問題)
        trans = text
        display_text = f"【系統警告：語意處理引擎暫時異常】\n\n{text}"

    # 🌟 2. 核心分析：後續邏輯必須用 trans 
    t_low = trans.lower()
    
    # ... (後面算分邏輯 hits, rule_bonus 等請保持不變) ...
    
    # ... (後面算分邏輯不變) ...
    
    reasons = []
    current_score = 0  # 🌟 關鍵修正：先初始化變數，避免 UnboundLocalError

    # --- 新增：信任因子（扣分制） ---
    trust_score = 0
    # 如果信件開頭很正式，或是結尾有標準退訂/版權宣告
    if any(w in t_low for w in ["dear customer", "regards", "copyright", "unsubscribe"]):
        trust_score -= 15
        reasons.append("✅ 包含正式商務禮儀或法律宣告 (減輕風險)")

    # 如果信件長度很長（詐騙通常短小精悍，講完重點就叫你點連結）
    if len(text) > 500:
        trust_score -= 10
        reasons.append("📝 內容敘述詳盡，與一般釣魚簡訊特徵不符 (減輕風險)")

    current_score += trust_score
    
    

    
    # 2. 結構化偵測
    links = re.findall(r'https?://([a-zA-Z0-9.-]+)', text)
    attachments = re.findall(r'\.(zip|exe|pdf|html|rar)', t_low)
    
    # 3. AI 原始分
    raw_prob_val = ai_model.predict_proba(tfidf_vec.transform([trans]))[0][1] * 100
    
    # --- 關鍵：將判斷理由與加權掛鉤 ---
    # A. 關鍵字加權
    # 找到 analyze_scam 函式中的加權部分並修改：

    # --- 優化後的加權邏輯 (避免分數暴走) ---
    
    # 1. 關鍵字遞減加權 (不再是死板的 15% * n)
    hits = [w for w in P_WEIGHTS[platform] if w in t_low or w in text]
    rule_bonus = 0
    if hits:
        u_hits = list(set(hits))
        rule_bonus = 10 + (len(u_hits) - 1) * 3
        rule_bonus = min(25, rule_bonus)
    
    current_score = (raw_prob_val * 0.4) + (rule_bonus * 0.6)
    if hits:
        reasons.append(f"🎯 偵測到風險關鍵字：{', '.join(u_hits)} (+{rule_bonus}%)")

    # B. 偵測加權 (0-100 單位)
    if attachments:
        current_score += 20
        reasons.append(f"偵測到可疑附件檔案 (*.{attachments[0]}) (+20%)")
    if links:
        current_score += 15
        reasons.append(f"🔗 包含可疑外部連結：`{links[0]}` (+15%)")
    if any(w in t_low for w in ["immediately", "3 days", "urgent", "立即", "三日內", "趕快"]):
        current_score += 10
        reasons.append("要求在限時內完成行動 (Urgency) (+10%)")

    # --- 🌟 核心：修正 80% 封頂邏輯 ---
    if not links and not attachments:
        if current_score > 80:
            current_score = 80.0
            reasons.insert(0, "⚠️ 未發現立即性惡意載體 (如連結/附件)，風險評等受限。")
        
    
    # 4. 判定類型
    # --- 4. 判定類型 (取代原本那段) ---
    scam_type = "一般威脅"
    
    # A. 針對帳務/發票
    if any(w in t_low for w in ["invoice", "payment", "overdue", "發票", "帳單", "欠費"]):
        scam_type = "帳務/發票詐騙"
    
    # B. 針對投資/獲利
    elif any(w in t_low for w in ["investment", "profit", "teacher", "飆股", "獲利", "群組"]):
        scam_type = "投資詐騙"
    
    # C. 針對包裹/簡訊
    elif any(w in t_low for w in ["package", "delivery", "包裹", "宅配", "超商"]):
        scam_type = "包裹/物流詐騙"
        
    # D. 針對帳號安全 (針對你目前的測試內容)
    elif any(w in t_low for w in ["suspended", "verify", "security", "login", "安全", "驗證", "凍結"]):
        scam_type = "帳據安全威脅"
    
    return {
        "final_score": max(0, min(current_score, 100.0)),
        "raw_prob": raw_prob_val,
        "explanations": reasons,
        "trans": display_text,  # <--- 確保這行是傳 display_text
        "type": scam_type,
        "detected_keywords": list(set(hits))
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
            s = res.get("final_score", 0)
            
            st.subheader("🕵️ 鑑定報告")
            st.metric("Scam Probability", f"{s:.2f}%", delta="🚨 HIGH" if s > 70 else "🟢 SAFE")
            # --- 🌟 決策組成分析 (垂直穩定版：取代原本的 columns) ---
            st.write("### ⚖️ 決策組成分析")
            raw_ai = res.get("raw_prob", 50)  # 取得 AI 原始分
            rule_weight = max(0, s - raw_ai)  # 計算規則加分
            
           # 在 col_res 顯示比重條的地方
            st.caption(f"🤖 AI 模型貢獻 (40% 比重)：{(raw_ai * 0.4):.1f}%")
            st.progress(min(1.0, (raw_ai * 0.4) / 40)) # 以 40 為分母

            if rule_weight > 0:
                st.caption(f"🛡️ 專家規則貢獻 (60% 比重)：{(rule_weight * 0.6):.1f}%")
                st.progress(min(1.0, (rule_weight * 0.6) / 60))
            
            st.write("---") # 分隔線，讓下方判斷原因更清晰

            # --- 1. 判斷原因 (這部分接在下方) ---
            # --- 修改後的判斷原因顯示區塊 ---
            st.write("### 📝 判斷原因")
            
            # 1. 先從 res 抓取資料，並給予預設值避免 NameError
            full_reasons = res.get("explanations", [])
            raw_ai_score = res.get("raw_prob", 0)  # 🌟 補上這一行！
            
            # 2. 開始顯示理由
            if full_reasons:
                for r in full_reasons:
                    clean_reason = r.split(' (+')[0] if ' (+' in r else r
                    st.markdown(f"* {clean_reason}")
            
            # 3. 執行剛才新增的 AI 貢獻邏輯
            if raw_ai_score > 60 and not full_reasons:
                st.markdown("* 🤖 **AI 語意模型偵測到高度異常模式。**")
            elif not full_reasons:
                st.write("🟢 未偵測到顯著風險特徵。")

            # 顯示判定類型
            # 找到顯示判定類型標籤的地方，改成這樣：
            s_type = res.get("type", "一般威脅")
            
            # 根據名稱給顏色：投資(紫), 帳務(橘), 安全(紅), 其他(藍)
            t_color = "#9333ea" if "投資" in s_type else \
                    ("#f97316" if "帳務" in s_type else \
                    ("#ef4444" if "安全" in s_type else "#3b82f6"))

            st.markdown(f"**判定類型：** <span class='platform-tag' style='background:{t_color}'>{s_type}</span>", unsafe_allow_html=True)

            # --- 2. AI 可解釋性分析 (XAI) (專業化：保留百分比) ---
            # --- 確保這幾行跟上面的 if full_reasons: 同一排垂直對齊 ---
            st.write("### 🧠 AI 可解釋性分析 (XAI)")
            if full_reasons:
                xai_content = "<br>".join([f"📈 {r}" for r in full_reasons])
                st.markdown(f'<div class="xai-box">{xai_content}</div>', unsafe_allow_html=True)
            else:
                st.info("💡 目前純依賴 AI 語意模型判定。")

            # 🌟 重點：這幾行必須「往左推」，跟上面的 if 同一排
            kws = res.get("detected_keywords", [])
            if kws:
                st.write("🔍 **系統特徵提取：**")
                # 修改 kw_html 的背景色
                kw_html = "".join([f"<span style='background:#E0F2FE; color:#0369A1; padding:3px 10px; border-radius:15px; margin-right:8px; font-size:0.85rem; border: 1px solid #7DD3FC;'>#{w}</span>" for w in kws])
                st.markdown(kw_html, unsafe_allow_html=True)
                st.write("") 

            with st.expander("📝 檢視語意處理結果"):
                raw_trans = res.get("trans", "") 
                platform_keywords = P_WEIGHTS.get(platform, [])
                for word in platform_keywords:
                    if word.lower() in raw_trans.lower():
                        import re
                        raw_trans = re.sub(f"({re.escape(word)})", r"**\1**", raw_trans, flags=re.IGNORECASE)
                st.markdown(raw_trans)
with tab2:
    st.subheader("📂 批量威脅鑑定中心")
    
    # 🌟 關鍵修復：先初始化變數，避免 NameError
    up_csv = None 
    
    # 然後才進行文件上傳
    up_csv = st.file_uploader("選擇上傳 CSV 檔案", type="csv", key="csv_file_uploader")
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