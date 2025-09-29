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
    This function is cached and is robust against empty or malformed dataframes.
    It now selectively merges columns to prevent duplication errors.
    """
    (rs_score_data, VP_change_D_data, VP_change_W_data, VP_change_M_data, VP_change_3M_data,
     source_52SET, source_52mai, perf_source_SET, perf_source_mai,
     SET_Stacked_bar, SET_VWAP, SET_VWAP_cumprod, mai_Stacked_bar, mai_VWAP, mai_VWAP_cumprod) = data_tuple

    # Start with the most comprehensive dataframe that includes market/sector info
    master_df = pd.DataFrame()
    if not rs_score_data.empty:
        master_df = rs_score_data.copy()

    # --- Prepare and merge other data sources one by one ---

    # 52 Week High/Low data
    source_52whl = pd.concat([source_52SET, source_52mai], ignore_index=True)
    if not source_52whl.empty and 'Name' in source_52whl.columns:
        cols_to_merge_52whl = ['Name', 'High', 'Low', 'Close', '52WHL', 'ROC_10']
        valid_cols = [col for col in cols_to_merge_52whl if col in source_52whl.columns]
        
        if not master_df.empty:
            master_df = pd.merge(master_df, source_52whl[valid_cols].drop_duplicates(subset=['Name']), on="Name", how="outer")
        else:
            master_df = source_52whl[valid_cols].drop_duplicates(subset=['Name'])

    # VP Change data - Selectively merge only the change columns
    for df, interval in [(VP_change_D_data, "D"), (VP_change_W_data, "W"), (VP_change_M_data, "M"), (VP_change_3M_data, "3M")]:
        if not df.empty and 'Symbols' in df.columns:
            df_renamed = df.rename(columns={"Symbols": "Name"})
            vp_cols_to_merge = ['Name', f'Price %Change {interval}', f'Volume %Change {interval}']
            valid_vp_cols = [col for col in vp_cols_to_merge if col in df_renamed.columns]
            
            if not master_df.empty:
                master_df = pd.merge(master_df, df_renamed[valid_vp_cols].drop_duplicates(subset=['Name']), on="Name", how="outer")
            else:
                 master_df = df_renamed[valid_vp_cols].drop_duplicates(subset=['Name'])

    # Performance data
    perf_list = []
    if not perf_source_SET.empty and perf_source_SET.shape[1] > 1: perf_list.append(perf_source_SET.iloc[:, [0, -1]])
    if not perf_source_mai.empty and perf_source_mai.shape[1] > 1: perf_list.append(perf_source_mai.iloc[:, [0, -1]])
    if perf_list:
        df_perf = pd.concat(perf_list, ignore_index=True)
        df_perf.columns = ["Name", "Performance 100 Days"]
        if not master_df.empty:
            master_df = pd.merge(master_df, df_perf.drop_duplicates(subset=['Name']), on="Name", how="outer")
        else:
            master_df = df_perf.drop_duplicates(subset=['Name'])
            
    # Clean up any merge-generated duplicate columns
    master_df = master_df.loc[:, ~master_df.columns.str.endswith('_x')]
    master_df = master_df.loc[:, ~master_df.columns.str.endswith('_y')]
    
    return master_df


def get_ai_response(prompt, data_for_ai_tuple):
    """
    Handles a general Q&A interaction with the Google Generative AI model.
    """
    if not GOOGLE_API_KEY:
        return "ข้อผิดพลาด: ไม่พบ Google API Key กรุณาตั้งค่าในไฟล์ .env"

    try:
        # --- [สำคัญ] แก้ไขชื่อโมเดลตรงนี้ ---
        if 'model' not in st.session_state:
            st.session_state.model = genai.GenerativeModel('gemini-1.5-flash-latest')

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

