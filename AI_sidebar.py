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
    This is the original, working version of the data preparation function.
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
    
    # Ensure index is reset before merging if it's not already a column
    if isinstance(SET_Stacked_bar.index, pd.RangeIndex):
        SET_Stacked_bar = SET_Stacked_bar.reset_index(names="Name")
    if isinstance(mai_Stacked_bar.index, pd.RangeIndex):
        mai_Stacked_bar = mai_Stacked_bar.reset_index(names="Name")


    # Combine and simplify data sources
    AIdata_52WHL = pd.concat([source_52SET, source_52mai])
    
    perf_list = []
    if not perf_source_SET.empty and perf_source_SET.shape[1] > 1: perf_list.append(perf_source_SET.iloc[:, [0, -1]])
    if not perf_source_mai.empty and perf_source_mai.shape[1] > 1: perf_list.append(perf_source_mai.iloc[:, [0, -1]])
    AIdata_Performance = pd.DataFrame(columns=["Name", "Performance 100 Days"])
    if perf_list:
        AIdata_Performance = pd.concat(perf_list, ignore_index=True)
        AIdata_Performance.columns = ["Name", "Performance 100 Days"]

    stack_vol_list = []
    if 'Name' in SET_Stacked_bar.columns and SET_Stacked_bar.shape[1] > 1: stack_vol_list.append(SET_Stacked_bar.iloc[:, [0, -1]])
    if 'Name' in mai_Stacked_bar.columns and mai_Stacked_bar.shape[1] > 1: stack_vol_list.append(mai_Stacked_bar.iloc[:, [0, -1]])
    AIdata_Stacked_Volume = pd.DataFrame(columns=["Name", "Volume Trade Ratio in Its Sector"])
    if stack_vol_list:
        AIdata_Stacked_Volume = pd.concat(stack_vol_list, ignore_index=True)
        AIdata_Stacked_Volume.columns = ["Name", "Volume Trade Ratio in Its Sector"]

    # Merge all data into a single DataFrame
    AIdata_ALL = rs_score_data.merge(AIdata_52WHL, how="outer", on="Name")
    AIdata_ALL = AIdata_ALL.merge(VP_change_D_data, how="outer", on="Name")
    AIdata_ALL = AIdata_ALL.merge(VP_change_W_data, how="outer", on="Name")
    AIdata_ALL = AIdata_ALL.merge(VP_change_M_data, how="outer", on="Name")
    AIdata_ALL = AIdata_ALL.merge(VP_change_3M_data, how="outer", on="Name")
    AIdata_ALL = AIdata_ALL.merge(AIdata_Performance, how="outer", on="Name")
    AIdata_ALL = AIdata_ALL.merge(AIdata_Stacked_Volume, how="outer", on="Name")
    
    # Clean up potential duplicate columns from merges
    AIdata_ALL = AIdata_ALL.loc[:,~AIdata_ALL.columns.duplicated()]

    return AIdata_ALL


def get_ai_response(prompt, data_for_ai_tuple, stock_name=None):
    """
    Handles a general Q&A interaction with the Google Generative AI model,
    combining the logic for both market summary and single stock analysis.
    """
    if not GOOGLE_API_KEY:
        return "ข้อผิดพลาด: ไม่พบ Google API Key กรุณาตั้งค่าในไฟล์ .env"

    try:
        # Use the model name from the working version
        if 'model' not in st.session_state:
            st.session_state.model = genai.GenerativeModel('gemini-1.5-flash')

        processed_data = prepare_ai_datasource(data_for_ai_tuple)
        
        if processed_data.empty:
            return "ไม่พบข้อมูลสำหรับวิเคราะห์ กรุณาตรวจสอบว่าข้อมูลถูกโหลดและประมวลผลอย่างถูกต้อง"

        context_df = pd.DataFrame()

        if stock_name:  # Logic for single stock analysis
            if stock_name in processed_data["Name"].values:
                stock_info_rows = processed_data[processed_data["Name"] == stock_name]
                if not stock_info_rows.empty:
                    stock_info = stock_info_rows.iloc[0]
                    sector = stock_info.get("sector")
                    market = stock_info.get("market")

                    if pd.notna(sector) and pd.notna(market):
                        context_df = processed_data[(processed_data["market"] == market) & (processed_data["sector"] == sector)]
                    else:
                        context_df = processed_data[processed_data["Name"] == stock_name]
                else: # Should not happen if stock_name is in values, but as a safeguard
                     return f"ไม่พบข้อมูลหุ้น '{stock_name}'"
            else:
                return f"ไม่พบหุ้น '{stock_name}' ในฐานข้อมูล"
        else:  # Logic for general market analysis
            context_df = processed_data.dropna(subset=['market', 'sector'])

        if context_df.empty:
            return "ไม่สามารถหาข้อมูลบริบทสำหรับสร้างคำตอบได้"
        
        csv_string = context_df.to_csv(index=False)

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

