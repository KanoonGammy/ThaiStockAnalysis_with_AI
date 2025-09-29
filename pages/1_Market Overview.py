# import necessary libraries
import pandas as pd
import streamlit as st
import datetime
import plotly.express as px
import numpy as np
import os # Import os to check for file existence

# --- [สำคัญ] แก้ไขการ import ให้เรียกจาก AI_sidebar.py ---
# Streamlit จะมองหาไฟล์นี้จากโฟลเดอร์หลัก (root) โดยอัตโนมัติ
from AI_sidebar import get_ai_response

# --- หมายเหตุ: เพื่อให้โค้ดสั้นกระชับ ผมจะจำลองฟังก์ชันบางตัวขึ้นมา ---
# ในโค้ดจริงของคุณ ให้ใช้ฟังก์ชันเดิมได้เลย
def select_market(tab_name):
    # Dummy function to simulate sidebar filters
    st.sidebar.header("Filter Options (Example)")
    st.sidebar.selectbox("Market", ["SET", "mai"], key=f"market_{tab_name}")
    st.sidebar.multiselect("Sector", ["TECH", "BANK", "ENERG"], key=f"sector_{tab_name}")
    # In your real app, this would return actual dataframes from select_market.py
    return pd.DataFrame({'symbol': ['ADVANC', 'AOT', 'BBL'], 'full name': ['advanc', 'aot', 'bbl']}), None, None, None, None

# --- เริ่มโค้ดแอปพลิเคชัน ---
st.set_page_config(layout="wide")

@st.cache_resource()
def import_data(period = 3650):
    # This is a simplified version for demonstration
    # In your real app, use your original complex data loading function
    dates = pd.to_datetime(pd.date_range(end=datetime.datetime.now(), periods=period))
    symbols = ['SET', 'ADVANC', 'AOT', 'BBL', 'CPALL', 'DELTA']
    price_data = np.random.rand(period, len(symbols)) * 100
    dp = pd.DataFrame(price_data, index=dates, columns=symbols)
    dp['SET'] = dp['SET'] * 20 + 1000 # Make SET index look more realistic
    
    volume_data = np.random.rand(period, len(symbols)) * 1_000_000
    dv = pd.DataFrame(volume_data, index=dates, columns=symbols)
    
    mc_data = {'Symbols': symbols[1:], 'MarketCap': np.random.rand(len(symbols)-1) * 1e12}
    mc = pd.DataFrame(mc_data)
    
    return dp, dv, mc

#raw standard
raw_std = pd.DataFrame({
    'symbol': ['ADVANC', 'AOT', 'BBL', 'CPALL', 'DELTA'],
    'market': ['SET', 'SET', 'SET', 'SET', 'SET'],
    'sector': ['TECH', 'TRANS', 'BANK', 'COMM', 'ETRON'],
    'sub-sector': ['ICT', 'AIR', 'BANKING', 'RETAIL', 'ELECTRONICS']
})


#import data
dp,dv,mc = import_data()

# --- UI Layout ---
hc1, hc2, hc3 = st.columns([7,7,2])
with hc1:
    st.header("APG Station: Market Overviews")
with hc3:
    SI = dp[["SET"]].copy().ffill().bfill()
    SI["Change"] = SI.pct_change()
    st.metric("SET Index", value = f"{SI['SET'].iloc[-1]:.2f}", delta = f"{SI['Change'].iloc[-1]:.4f}%" )

# --- Tabs for content and AI ---
# We will add the AI Q&A as a final section instead of a tab for better visibility
st.info("เนื้อหาส่วนนี้เป็นข้อมูลตัวอย่างเพื่อแสดงการทำงานร่วมกับ AI Q&A ด้านล่าง")

tab1, tab2, tab3 = st.tabs(["🏆 Ranking", "📈 Performance", "📊 Volume"])

with tab1:
    st.subheader("ตัวอย่างข้อมูล Ranking")
    st.write("ในแอปพลิเคชันจริง ส่วนนี้จะแสดงผลการวิเคราะห์ Ranking และ Price/Volume Change")
    # For demonstration, we'll create dummy dataframes that the AI needs
    rs_score_data = pd.DataFrame({'Name': ['ADVANC', 'AOT'], 'RS Score': [0.8, 0.6], 'market': ['SET','SET'], 'sector':['TECH','TRANS']})
    VP_change_D_data = pd.DataFrame({'Symbols': ['ADVANC', 'AOT'], 'Price %Change D': [1.5, -0.5]})
    VP_change_W_data = pd.DataFrame({'Symbols': ['ADVANC', 'AOT'], 'Price %Change W': [3.2, 1.1]})
    VP_change_M_data = pd.DataFrame({'Symbols': ['ADVANC', 'AOT'], 'Price %Change M': [5.0, 2.5]})
    VP_change_3M_data = pd.DataFrame({'Symbols': ['ADVANC', 'AOT'], 'Price %Change 3M': [10.0, 8.2]})
with tab2:
    st.subheader("ตัวอย่างข้อมูล Performance")
    source_52SET = pd.DataFrame({'Name': ['ADVANC', 'AOT'], '52WHL': [0.9, 0.7]})
    source_52mai = pd.DataFrame()
    perf_source_SET = pd.DataFrame({'Name': ['ADVANC', 'AOT'], 'Return': [1.1, 1.08]})
    perf_source_mai = pd.DataFrame()
with tab3:
    st.subheader("ตัวอย่างข้อมูล Volume")
    SET_Stacked_bar = pd.DataFrame(index=['ADVANC', 'AOT'], data={'VolumeRatio': [0.6, 0.4]})
    SET_VWAP = pd.DataFrame({'Name': ['ADVANC', 'AOT'], 'VWAP': [150.5, 65.2]})
    SET_VWAP_cumprod = pd.DataFrame({'Name': ['ADVANC', 'AOT'], 'Cumprod': [1.2, 1.15]})
    mai_Stacked_bar = pd.DataFrame()
    mai_VWAP = pd.DataFrame()
    mai_VWAP_cumprod = pd.DataFrame()


st.divider()
# --- AI Q&A Section moved to the main page body ---
st.header("🤖 AI Q&A Section")
st.markdown("ถามคำถามเกี่ยวกับภาพรวมตลาดหรือหุ้นรายตัวจากข้อมูลที่แอปพลิเคชันวิเคราะห์")
ai_response_placeholder = st.container()

# --- Sidebar for Filters and AI Input ---
# Sidebar is processed last to ensure all dataframes are created
with st.sidebar:
    # --- Filter Section ---
    # In your real app, you would import and use the actual select_market function
    select_market("main_filters")

    st.divider()
    # --- AI Q&A Input Section ---
    st.header("🤖 AI Q&A")
    user_question = st.text_area("ป้อนคำถามของคุณที่นี่...",
                                 value="หุ้นตัวไหนมี RS_Rank ดีที่สุด และมีแนวโน้มเป็นอย่างไร?",
                                 height=150,
                                 key="ai_question")

    if st.button("ส่งคำถามถึง AI", key="ask_ai_button"):
        if not user_question.strip():
            st.warning("กรุณาป้อนคำถาม")
        else:
            # Prepare the data tuple for the AI
            data_for_ai_tuple = (
                rs_score_data, VP_change_D_data, VP_change_W_data, VP_change_M_data, VP_change_3M_data,
                source_52SET, source_52mai, perf_source_SET, perf_source_mai,
                SET_Stacked_bar, SET_VWAP, SET_VWAP_cumprod,
                mai_Stacked_bar, mai_VWAP, mai_VWAP_cumprod
            )

            generic_prompt = f"""
                **Persona:** You are a highly skilled Senior Financial Analyst. Your sole purpose is to answer the user's question based *strictly and exclusively* on the technical data provided in the attached CSV file. Do not use any external knowledge.

                **User's Question:** "{user_question}"

                **Core Task:**
                1.  Thoroughly analyze the provided CSV data to find information relevant to the user's question.
                2.  Formulate a concise, data-driven answer in Thai.
                3.  **Crucially, you must back up every claim with specific data points from the file.** For example, instead of saying "The stock is strong," say "หุ้น A แสดงโมเมนตัมที่แข็งแกร่ง, เห็นได้จากค่า RS_Rank ที่ X และ Price %Change D ที่ Y%."
                4.  If data required is missing, explicitly state that it's unavailable.
            """
            response_text = get_ai_response(generic_prompt, data_for_ai_tuple)

            with ai_response_placeholder:
                # Clear previous response and show the new one
                ai_response_placeholder.empty()
                st.markdown("### 🤖 คำตอบจาก AI")
                st.markdown("---")
                st.markdown(response_text)

