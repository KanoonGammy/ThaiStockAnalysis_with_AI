# import necessary libraries
import streamlit as st
import openai # Import OpenAI library
import pandas as pd
import os
import time
from dotenv import load_dotenv

# Load environment variables for the API key
load_dotenv()
# --- [MODIFIED] Load OpenAI API Key ---
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# It's good practice to also check for the key existence here
if not OPENAI_API_KEY:
    st.warning("OpenAI API Key not found. Please set it in your .env file for AI analysis to work.")

@st.cache_data
def prepare_ai_datasource(data_tuple):
    """
    Prepares and merges all necessary data sources into a single DataFrame for AI analysis.
    This function is cached by Streamlit to avoid re-computation.
    """
    # Clean all dataframes in the tuple
    cleaned_tuple = []
    for df in data_tuple:
        if isinstance(df, pd.DataFrame):
            unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
            cleaned_tuple.append(df.drop(columns=unnamed_cols) if unnamed_cols else df)
        else:
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
    
    if not SET_Stacked_bar.empty: SET_Stacked_bar = SET_Stacked_bar.reset_index(names="Name")
    if not mai_Stacked_bar.empty: mai_Stacked_bar = mai_Stacked_bar.reset_index(names="Name")


    # Combine and simplify data sources
    AIdata_52WHL = pd.concat([source_52SET, source_52mai], ignore_index=True)
    
    perf_list = []
    if not perf_source_SET.empty and perf_source_SET.shape[1] > 1: perf_list.append(perf_source_SET.iloc[:, [0, -1]])
    if not perf_source_mai.empty and perf_source_mai.shape[1] > 1: perf_list.append(perf_source_mai.iloc[:, [0, -1]])
    AIdata_Performance = pd.DataFrame(columns=["Name", "Performance 100 Days"])
    if perf_list:
        AIdata_Performance = pd.concat( perf_list, ignore_index=True)
        AIdata_Performance.columns = ["Name", "Performance 100 Days"]

    stack_vol_list = []
    if 'Name' in SET_Stacked_bar.columns and SET_Stacked_bar.shape[1] > 1: stack_vol_list.append(SET_Stacked_bar.iloc[:, [0, -1]])
    if 'Name' in mai_Stacked_bar.columns and mai_Stacked_bar.shape[1] > 1: stack_vol_list.append(mai_Stacked_bar.iloc[:, [0, -1]])
    AIdata_Stacked_Volume = pd.DataFrame(columns=["Name", "Volume Trade Ratio in Its Sector"])
    if stack_vol_list:
        AIdata_Stacked_Volume = pd.concat(stack_vol_list, ignore_index=True)
        AIdata_Stacked_Volume.columns = ["Name", "Volume Trade Ratio in Its Sector"]

    # --- [FIXED] Merge all data into a single DataFrame ---
    AIdata_ALL = rs_score_data.merge(AIdata_52WHL, how="outer", on="Name", suffixes=('_rs', '_52'))

    for col in ['market', 'sector', 'sub-sector']:
        col_rs, col_52 = f'{col}_rs', f'{col}_52'
        if col_rs in AIdata_ALL.columns and col_52 in AIdata_ALL.columns:
            AIdata_ALL[col] = AIdata_ALL[col_rs].fillna(AIdata_ALL[col_52])
            AIdata_ALL = AIdata_ALL.drop(columns=[col_rs, col_52])

    dfs_to_merge = [
        VP_change_D_data, VP_change_W_data, VP_change_M_data, VP_change_3M_data,
        AIdata_Performance, AIdata_Stacked_Volume
    ]
    for df in dfs_to_merge:
        if not df.empty and 'Name' in df.columns:
             AIdata_ALL = AIdata_ALL.merge(df, how="outer", on="Name")

    AIdata_ALL = AIdata_ALL.loc[:,~AIdata_ALL.columns.duplicated()]
    return AIdata_ALL


def get_ai_response(prompt, market_filter="All", sector_filter="All", sub_sector_filter="All"):
    """
    Handles a general Q&A interaction with the OpenAI API.
    """
    if not OPENAI_API_KEY:
        return "ข้อผิดพลาด: ไม่พบ OpenAI API Key กรุณาตั้งค่าในไฟล์ .env"

    try:
        # --- [MODIFIED] Initialize OpenAI client ---
        client = openai.OpenAI(api_key=OPENAI_API_KEY)

        processed_data = st.session_state.get('ai_datasource')
        if processed_data is None or processed_data.empty:
            return "ไม่พบข้อมูลสำหรับวิเคราะห์ กรุณารอให้ข้อมูลโหลดเสร็จสมบูรณ์"

        # Filter data based on user selection
        context_df = processed_data.copy()
        if market_filter != "All":
            context_df = context_df[context_df['market'] == market_filter]
        if sector_filter != "All":
            context_df = context_df[context_df['sector'] == sector_filter]
        if sub_sector_filter != "All":
            context_df = context_df[context_df['sub-sector'] == sub_sector_filter]

        if context_df.empty:
            return "ไม่พบข้อมูลหุ้นตามเงื่อนไขที่คุณเลือก"

        csv_string = context_df.to_csv(index=False)

        # --- [MODIFIED] API call to OpenAI ---
        # Create a combined message for the user role
        full_user_prompt = f"""
        Here is the user's request:
        ---
        {prompt}
        ---

        Analyze the following CSV data to answer the request:
        ---
        {csv_string}
        ---
        """

        with st.spinner("OpenAI กำลังวิเคราะห์..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert AI technical analyst for the stock market. Your entire analysis must be based *only* on the provided CSV data."},
                    {"role": "user", "content": full_user_prompt}
                ]
            )
            return response.choices[0].message.content

    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดระหว่างการวิเคราะห์ของ AI: {e}")
        return f"ขออภัย, เกิดข้อผิดพลาดในการวิเคราะห์: {e}"

