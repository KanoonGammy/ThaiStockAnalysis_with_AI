# import necessary libraries
import streamlit as st
import google.generativeai as genai
import pandas as pd
import os
import time # Import the time module for handling delays
from dotenv import load_dotenv

# Load environment variables for the API key
# โหลดค่า API Key จากไฟล์ .env
load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# Configure the generative AI model with the API key
# ตั้งค่า API Key หากมี
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    # แสดงคำเตือนหากไม่พบ API Key
    st.warning("Google API Key not found. Please set it in your environment variables for AI analysis to work.")


@st.cache_data
def prepare_ai_datasource(data_tuple):
    """
    Prepares and merges all necessary data sources into a single DataFrame for AI analysis.
    This function is cached to avoid re-computation.
    It takes a tuple of DataFrames as input.
    """
    # Clean all dataframes in the tuple by removing 'Unnamed' columns which can cause merge errors
    cleaned_tuple = []
    for df in data_tuple:
        if isinstance(df, pd.DataFrame):
            # Find and drop any columns that start with 'Unnamed:'
            unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
            cleaned_tuple.append(df.drop(columns=unnamed_cols) if unnamed_cols else df)
        else:
            # Keep non-DataFrame elements as they are
            cleaned_tuple.append(df)

    # Unpack the cleaned dataframes
    (rs_score_data, VP_change_D_data, VP_change_W_data, VP_change_M_data, VP_change_3M_data,
     source_52SET, source_52mai, perf_source_SET, perf_source_mai,
     SET_Stacked_bar, SET_VWAP, SET_VWAP_cumprod, mai_Stacked_bar, mai_VWAP, mai_VWAP_cumprod) = tuple(cleaned_tuple)


    # Rename columns for consistency
    VP_change_D_data = VP_change_D_data.rename(columns={"Symbols": "Name"})
    VP_change_W_data = VP_change_W_data.rename(columns={"Symbols": "Name"}).drop("MarketCap", axis=1, errors='ignore')
    VP_change_M_data = VP_change_M_data.rename(columns={"Symbols": "Name"}).drop("MarketCap", axis=1, errors='ignore')
    VP_change_3M_data = VP_change_3M_data.rename(columns={"Symbols": "Name"}).drop("MarketCap", axis=1, errors='ignore')
    SET_Stacked_bar = SET_Stacked_bar.reset_index(names="Name")
    mai_Stacked_bar = mai_Stacked_bar.reset_index(names="Name")

    # Combine and simplify data sources
    AIdata_52WHL = pd.concat([source_52SET, source_52mai])
    AIdata_Performance = pd.concat([perf_source_SET.iloc[:, [0, -1]], perf_source_mai.iloc[:, [0, -1]]])
    AIdata_Performance.columns = ["Name", "Performance 100 Days"]
    AIdata_Stacked_Volume = pd.concat([SET_Stacked_bar.iloc[:, [0, -1]], mai_Stacked_bar.iloc[:, [0, -1]]])
    AIdata_Stacked_Volume.columns = ["Name", "Volume Trade Ratio in Its Sector"]
    AIdata_VWAP = pd.concat([SET_VWAP.iloc[:, [0, -1]], mai_VWAP.iloc[:, [0, -1]]])
    AIdata_VWAP.columns = ["Name", "Volume Weighted Average Price"]
    AIdata_Cumulative_VWAP = pd.concat([SET_VWAP_cumprod.iloc[:, [0, -1]], mai_VWAP_cumprod.iloc[:, [0, -1]]])
    AIdata_Cumulative_VWAP.columns = ["Name", "Cumulative_VWAP"]

    # Merge all data into a single DataFrame
    AIdata_ALL = rs_score_data.merge(AIdata_52WHL, how="outer", on="Name")
    AIdata_ALL = AIdata_ALL.merge(VP_change_D_data, how="outer", on="Name")
    AIdata_ALL = AIdata_ALL.merge(VP_change_W_data, how="outer", on="Name")
    AIdata_ALL = AIdata_ALL.merge(VP_change_M_data, how="outer", on="Name")
    AIdata_ALL = AIdata_ALL.merge(VP_change_3M_data, how="outer", on="Name")
    AIdata_ALL = AIdata_ALL.merge(AIdata_Performance, how="outer", on="Name")
    AIdata_ALL = AIdata_ALL.merge(AIdata_Stacked_Volume, how="outer", on="Name")
    AIdata_ALL = AIdata_ALL.merge(AIdata_VWAP, how="outer", on="Name")
    AIdata_ALL = AIdata_ALL.merge(AIdata_Cumulative_VWAP, how="outer", on="Name")

    return AIdata_ALL


def AI_Overviews(prompt, data_for_ai_tuple, stock_name):
    """
    Handles the interaction with the Google Generative AI model.
    NOW RETURNS the response text or an error message.
    """
    if not GOOGLE_API_KEY:
        st.error("AI analysis cannot proceed. Google API Key is missing.")
        return # หยุดการทำงานและไม่คืนค่าอะไร

    try:
        # ... (ส่วนของการเตรียมข้อมูลเหมือนเดิม) ...
        # st.session_state.model = genai.GenerativeModel(...)
        # processed_data = prepare_ai_datasource(...)

        # --- ส่วนที่เหลือของ Logic เหมือนเดิม จนถึงการเรียก AI ---
        
        # --- Exponential Backoff Logic ---
        max_retries = 5
        base_delay = 5
        for attempt in range(max_retries):
            try:
                with st.spinner(f"AI กำลังวิเคราะห์ข้อมูล... (Attempt {attempt + 1}/{max_retries})"):
                    response = st.session_state.model.generate_content([prompt, csv_string])
                    return response.text  # <<<<<<< [สำคัญ] เปลี่ยนจาก st.write เป็น return

            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    st.warning(f"Rate limit reached. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    st.error(f"An error occurred during AI analysis: {e}")
                    return f"เกิดข้อผิดพลาดในการวิเคราะห์: {e}" # คืนค่าเป็น Error message
        
        return "ไม่สามารถเชื่อมต่อกับ AI ได้หลังจากการพยายามหลายครั้ง" # กรณี Retry จนหมด

    except Exception as e:
        st.error(f"An error occurred during data preparation for AI: {e}")
        return f"เกิดข้อผิดพลาดในการเตรียมข้อมูล: {e}" # คืนค่าเป็น Error message

# --- ทำเช่นเดียวกันกับฟังก์ชัน AI_Market_Summary ---

def AI_Market_Summary(prompt, data_for_ai_tuple):
    """
    Handles generating a market-wide summary using the AI model.
    NOW RETURNS the response text or an error message.
    """
    if not GOOGLE_API_KEY:
        st.error("AI analysis cannot proceed. Google API Key is missing.")
        return

    try:
        # ... (ส่วนของการเตรียมข้อมูลเหมือนเดิม) ...
        
        # --- Exponential Backoff Logic ---
        max_retries = 5
        base_delay = 5
        for attempt in range(max_retries):
            try:
                with st.spinner(f"AI กำลังสร้างบทสรุปภาพรวมตลาด... (Attempt {attempt + 1}/{max_retries})"):
                    response = st.session_state.model.generate_content([prompt, csv_string])
                    return response.text  # <<<<<<< [สำคัญ] เปลี่ยนจาก st.write เป็น return
                    
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    st.warning(f"Rate limit reached. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    st.error(f"An error occurred during AI market summary generation: {e}")
                    return f"เกิดข้อผิดพลาดในการสร้างบทสรุป: {e}"

        return "ไม่สามารถเชื่อมต่อกับ AI ได้หลังจากการพยายามหลายครั้ง"

    except Exception as e:
        st.error(f"An error occurred during data preparation for market summary: {e}")
        return f"เกิดข้อผิดพลาดในการเตรียมข้อมูล: {e}"
