import streamlit as st
import pandas as pd
import joblib
import os
import re
from deep_translator import GoogleTranslator
from langdetect import detect

# -----------------------------
# 1. 頁面設定
# -----------------------------
st.set_page_config(
    page_title="AI Phishing Guard Pro",
    layout="wide",
    page_icon="🛡️"
)

# -----------------------------
# 2. CSS UI 優化
# -----------------------------
st.markdown("""
<style>
.stApp { background-color: #ffffff; color: #1f2937; }
[data-testid="stSidebar"] { background-color: #f9fafb; }

.metric-card {
background: #ffffff;
border-radius: 12px;
padding: 20px;
border: 1px solid #e5e7eb;
border-left: 6px solid #3b82f6;
margin-bottom: 15px;
}

.structure-box {
background-color: #f8fafc;
border-radius: 8px;
padding: 12px;
border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 3. 側邊欄
# -----------------------------
with st.sidebar:

    st.title("🛡️ Threat Intelligence")

    st.markdown("""
    **Dataset Size**  
    40,000 emails
    
    **Model**  
    Multinomial Naive Bayes
    
    **Feature**  
    TF-IDF
    
    **Accuracy**  
    92%
    """)

    img_path = "analysis_result.png"

    if os.path.exists(img_path):
        st.image(img_path, caption="Behavior Baseline Analysis")

# -----------------------------
# 4. 載入模型
# -----------------------------
@st.cache_resource
def load_model():

    try:

        tfidf = joblib.load("tfidf.pkl")
        model = joblib.load("model.pkl")

        return tfidf, model

    except:

        st.error("找不到模型檔案 model.pkl / tfidf.pkl")
        st.stop()

tfidf_vec, ai_model = load_model()

# -----------------------------
# 5. 偵測功能
# -----------------------------

urgency_words = {
"urgent","immediately","verify","suspended",
"final warning","expired","limited"
}

finance_words = {
"bank","payment","login","account",
"invoice","credentials"
}

def translate_if_needed(text):

    try:

        lang = detect(text)

        if lang != "en":
            return GoogleTranslator(source='auto', target='en').translate(text)

        return text

    except:

        return text


def detect_links(text):

    return re.findall(r'https?://\S+', text)


def detect_keywords(text, keywords):

    hits = []

    lower = text.lower()

    for w in keywords:

        if w in lower:

            hits.append(w)

    return hits


def calculate_risk(prob, links, urg, fin):

    score = prob

    score += 0.1 * len(links)

    score += 0.15 * len(urg)

    score += 0.15 * len(fin)

    return min(score, 1)


def analyze_email(text):

    translated = translate_if_needed(text)

    vec = tfidf_vec.transform([translated])

    prob = ai_model.predict_proba(vec)[0][1]

    links = detect_links(text)

    urg_hits = detect_keywords(translated, urgency_words)

    fin_hits = detect_keywords(translated, finance_words)

    risk = calculate_risk(prob, links, urg_hits, fin_hits)

    return {
        "risk": risk,
        "links": links,
        "urg": urg_hits,
        "fin": fin_hits,
        "translated": translated
    }

# -----------------------------
# 6. 主頁
# -----------------------------
st.title("🛡️ AI Phishing Detection Platform")

tab1, tab2 = st.tabs(["Single Email Scan","CSV Batch Scan"])

# -----------------------------
# TAB 1
# -----------------------------
with tab1:

    col1, col2 = st.columns([1.2,1])

    with col1:

        st.subheader("Email Input")

        example_phishing = """
Dear Customer,

Your account has been suspended.

Click the link below to verify your account immediately.

http://fake-bank-login.com
"""

        if st.button("Load Phishing Example"):

            st.session_state.email = example_phishing

        user_input = st.text_area(
            "Paste email content",
            height=300,
            key="email"
        )

        if st.button("Analyze Email"):

            if user_input:

                result = analyze_email(user_input)

                st.session_state.result = result

            else:

                st.warning("Please input email text")

    with col2:

        if "result" in st.session_state:

            r = st.session_state.result

            score = r["risk"] * 100

            if score > 70:

                label = "🔴 HIGH RISK"

            elif score >= 40:

                label = "🟡 MEDIUM RISK"

            else:

                label = "🟢 SAFE"

            st.subheader("Threat Score")

            st.metric("Risk Level", f"{score:.2f}%", label)

            st.progress(score/100)

            st.write("### Structure Analysis")

            st.markdown(f"""
<div class="structure-box">

Links detected: **{len(r['links'])}**

Urgency language: **{'YES' if r['urg'] else 'NO'}**

Financial request: **{'YES' if r['fin'] else 'NO'}**

</div>
""", unsafe_allow_html=True)

            st.write("### Suspicious Keywords")

            hits = list(set(r["urg"] + r["fin"]))

            if hits:

                st.warning(", ".join(hits))

            else:

                st.success("No strong phishing indicators")

            with st.expander("Translated Text"):

                st.write(r["translated"])

# -----------------------------
# TAB 2
# -----------------------------
with tab2:

    st.subheader("Batch Analysis")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        text_col = st.selectbox("Select text column", df.columns)

        limit = st.slider("Analyze rows", 1, 1000, 50)

        if st.button("Start Scan"):

            results = []

            for txt in df[text_col].astype(str).tolist()[:limit]:

                r = analyze_email(txt)

                results.append(r["risk"])

            df["risk_score"] = results + [None]*(len(df)-len(results))

            st.dataframe(df.head(limit))

            st.download_button(
                "Download Result CSV",
                df.to_csv(index=False),
                file_name="phishing_analysis.csv"
            )

            st.success("Batch analysis completed")