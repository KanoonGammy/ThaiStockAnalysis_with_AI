# import necessary libraries
import streamlit as st
import google.generativeai as genai
import pandas as pd
import os
import time # Import the time module for handling delays
from dotenv import load_dotenv

# Load environment variables for the API key
load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Configure the generative AI model with the API key
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.warning("Google API Key not found. Please set it in your environment variables for AI analysis to work.")


@st.cache_data
def prepare_ai_datasource(data_tuple):
    """
    Prepares and merges all necessary data sources into a single DataFrame for AI analysis.
    This function is cached to avoid re-computation and is now more robust against empty dataframes.
    """
    # Unpack the tuple
    (rs_score_data, VP_change_D_data, VP_change_W_data, VP_change_M_data, VP_change_3M_data,
     source_52SET, source_52mai, perf_source_SET, perf_source_mai,
     SET_Stacked_bar, SET_VWAP, SET_VWAP_cumprod, mai_Stacked_bar, mai_VWAP, mai_VWAP_cumprod) = data_tuple

    # --- Robustly combine and simplify data sources ---
    AIdata_52WHL = pd.concat([source_52SET, source_52mai], ignore_index=True) if not source_52SET.empty or not source_52mai.empty else pd.DataFrame()

    perf_list = []
    if not perf_source_SET.empty and perf_source_SET.shape[1] > 1:
        perf_list.append(perf_source_SET.iloc[:, [0, -1]])
    if not perf_source_mai.empty and perf_source_mai.shape[1] > 1:
        perf_list.append(perf_source_mai.iloc[:, [0, -1]])
    if perf_list:
        AIdata_Performance = pd.concat(perf_list, ignore_index=True)
        AIdata_Performance.columns = ["Name", "Performance 100 Days"]
    else:
        AIdata_Performance = pd.DataFrame(columns=["Name", "Performance 100 Days"])

    stack_vol_list = []
    if not SET_Stacked_bar.empty and SET_Stacked_bar.shape[1] > 1:
        stack_vol_list.append(SET_Stacked_bar.reset_index().iloc[:, [0, -1]])
    if not mai_Stacked_bar.empty and mai_Stacked_bar.shape[1] > 1:
        stack_vol_list.append(mai_Stacked_bar.reset_index().iloc[:, [0, -1]])
    if stack_vol_list:
        AIdata_Stacked_Volume = pd.concat(stack_vol_list, ignore_index=True)
        AIdata_Stacked_Volume.columns = ["Name", "Volume Trade Ratio in Its Sector"]
    else:
        AIdata_Stacked_Volume = pd.DataFrame(columns=["Name", "Volume Trade Ratio in Its Sector"])
        
    vwap_list = []
    if not SET_VWAP.empty and SET_VWAP.shape[1] > 1:
        vwap_list.append(SET_VWAP.iloc[:, [0, -1]])
    if not mai_VWAP.empty and mai_VWAP.shape[1] > 1:
        vwap_list.append(mai_VWAP.iloc[:, [0, -1]])
    if vwap_list:
        AIdata_VWAP = pd.concat(vwap_list, ignore_index=True)
        AIdata_VWAP.columns = ["Name", "Volume Weighted Average Price"]
    else:
        AIdata_VWAP = pd.DataFrame(columns=["Name", "Volume Weighted Average Price"])

    cum_vwap_list = []
    if not SET_VWAP_cumprod.empty and SET_VWAP_cumprod.shape[1] > 1:
        cum_vwap_list.append(SET_VWAP_cumprod.iloc[:, [0, -1]])
    if not mai_VWAP_cumprod.empty and mai_VWAP_cumprod.shape[1] > 1:
        cum_vwap_list.append(mai_VWAP_cumprod.iloc[:, [0, -1]])
    if cum_vwap_list:
        AIdata_Cumulative_VWAP = pd.concat(cum_vwap_list, ignore_index=True)
        AIdata_Cumulative_VWAP.columns = ["Name", "Cumulative_VWAP"]
    else:
        AIdata_Cumulative_VWAP = pd.DataFrame(columns=["Name", "Cumulative_VWAP"])

    # List of all dataframes to merge
    dfs_to_merge = [
        rs_score_data, AIdata_52WHL, 
        VP_change_D_data.rename(columns={"Symbols":"Name"}), 
        VP_change_W_data.rename(columns={"Symbols":"Name"}), 
        VP_change_M_data.rename(columns={"Symbols":"Name"}), 
        VP_change_3M_data.rename(columns={"Symbols":"Name"}),
        AIdata_Performance, AIdata_Stacked_Volume, AIdata_VWAP, AIdata_Cumulative_VWAP
    ]
    
    # Start with a base dataframe if available
    AIdata_ALL = pd.DataFrame()
    if not rs_score_data.empty:
        AIdata_ALL = rs_score_data.copy()

    # Iteratively merge other dataframes
    for df in dfs_to_merge[1:]:
        if not df.empty and 'Name' in df.columns:
            if not AIdata_ALL.empty:
                AIdata_ALL = pd.merge(AIdata_ALL, df, on="Name", how="outer")
            else:
                AIdata_ALL = df.copy() # If base was empty, start with the first valid df

    return AIdata_ALL


def get_ai_response(prompt, data_for_ai_tuple):
    """
    Handles a general Q&A interaction with the Google Generative AI model.
    """
    if not GOOGLE_API_KEY:
        return "ข้อผิดพลาด: ไม่พบ Google API Key กรุณาตั้งค่าในไฟล์ .env"

    try:
        if 'model' not in st.session_state:
            st.session_state.model = genai.GenerativeModel('gemini-1.5-flash')

        processed_data = prepare_ai_datasource(data_for_ai_tuple)
        
        if processed_data.empty:
            return "ไม่พบข้อมูลสำหรับวิเคราะห์ กรุณาตรวจสอบว่าข้อมูลถูกโหลดและประมวลผลอย่างถูกต้องในแต่ละ Tab"

        csv_string = processed_data.to_csv(index=False)

        max_retries = 5
        base_delay = 5
        for attempt in range(max_retries):
            try:
                with st.spinner(f"AI กำลังประมวลผล... (ครั้งที่ {attempt + 1}/{max_retries})"):
                    response = st.session_state.model.generate_content([prompt, csv_string])
                    return response.text
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    st.warning(f"Rate limit reached. กำลังลองใหม่ในอีก {delay} วินาที...")
                    time.sleep(delay)
                else:
                    st.error(f"เกิดข้อผิดพลาดระหว่างการวิเคราะห์ของ AI: {e}")
                    return f"ขออภัย, เกิดข้อผิดพลาดในการวิเคราะห์: {e}"

        return "ไม่สามารถเชื่อมต่อกับ AI ได้หลังจากการพยายามหลายครั้ง"

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการเตรียมข้อมูลสำหรับ AI: {e}")
        return f"ขออภัย, เกิดข้อผิดพลาดในการเตรียมข้อมูล: {e}"

