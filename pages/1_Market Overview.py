# import necessary libraries
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_ta as ta
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import time
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from select_market import select_market
# Import the new generic AI function
from AI_module import get_ai_response

st.set_page_config(layout="wide")

#https://docs.streamlit.io/develop/api-reference/data/st.metric
#"""===================================IMPORT DATA==================================="""
@st.cache_resource()
def import_data(period = 3650):
    # import data price
    import pandas as pd

    # 1. สร้าง URL สำหรับดาวน์โหลดเป็นไฟล์ CSV
    sheet_id = "1Uk2bEql-MDVuEzka4f1Y7bMBM_vbob7mXb75ITUwwII"  # << ใส่ ID ของชีท (ส่วนที่อยู่ใน URL)
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

    # 2. อ่านข้อมูลจาก URL ด้วย pandas และเลือกเฉพาะท้ายตารางตามตัวแปร period
    dp = pd.read_csv(sheet_url).tail(period)

    # 3. แปลงคอลัมน์ Date ให้เป็น Datetime และตั้งเป็น Index (เหมือนเดิม)
    dp['Date'] = pd.to_datetime(dp["Date"])
    dp = dp.set_index("Date")

# ตอนนี้ dp ก็พร้อมใช้งานแล้ว

    # rename columns
    t = [x.split(".")[0] for x in dp.columns] # list comprehesion >> split .BK
    index_set = t.index("^SET") # find ^SET index
    t[index_set] = "SET" # rename ^SET as SET
    dp.columns = t # rename index

    # clean data
    pass

    # import data volume
    # Assuming source_volume.csv and source_mkt.csv are in the same directory
    try:
        dv = pd.read_csv("source_volume.csv").tail(3650)
        dv['Date'] = pd.to_datetime(dv['Date'])
        dv = dv.set_index("Date")
        dv.columns = t #use columns dp

        # import data marketcap
        mc = pd.read_csv("source_mkt.csv")
        mc = mc.set_index(mc.columns[1]).T
        m = [x.split(".")[0] for x in mc.columns]
        mc.columns = m
        mc = mc.T.reset_index().rename(columns = {"index": "Symbols"})
        mc["MarketCap"] = mc["MarketCap"].astype(float).fillna(0)
    except FileNotFoundError:
        st.error("Error: 'source_volume.csv' or 'source_mkt.csv' not found. Please make sure they are in the correct directory.")
        st.stop()


    return dp,dv,mc

#import data
dp,dv,mc = import_data()
#raw standard
raw_std = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET_Index.csv", index_col= "Unnamed: 0").sort_values(["symbol"], ascending= True)
raw_set50 = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET50.csv")['symbol'].tolist()
raw_set100 = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET100.csv")['symbol'].tolist()

hc1,hc2,hc3 = st.columns([7,7,2])
with hc1:
    st.header("APG Station: Market Overviews")
    pass
#"""===================================Count Change==================================="""
with hc2: # พักเก็บ counting daily
    pass

with hc3:
    #"""===================================SHOW SET INDEX==================================="""
    SI = dp[["SET"]].copy().ffill().bfill()
    SI["Change"] = SI.pct_change()
    st.metric("SET Index", value = f"{SI['SET'].iloc[-1]:.2f}", delta = f"{SI['Change'].iloc[-1]:.4f}%" )

#"""===================================SETTING TAB==================================="""
tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking and Change", "📈 Performance and 52 Week High Low",
                                 "📊 Volume Analysis", "🔥 Sector Heatmap"])

# ... (โค้ดทั้งหมดของ tab1, tab2, tab3, tab4 เหมือนเดิมทุกประการ) ...
# The code for all the tabs (tab1, tab2, tab3, tab4) remains unchanged.
# I am omitting it here for brevity but you should keep it in your file.
with tab1:
    #"""===================================SETTING Sidebar==================================="""
    selection = None
    # with st.sidebar:
    rawt1,sl11,sl12,sl13,sl14 = select_market("tab1")
    rawt1["sub-sector"] = rawt1["sub-sector"].fillna("None")

    selection = rawt1.reset_index(drop = True)
    selection.index = selection.index + 1
    selection = selection["symbol"]

# create columns
    col0, col1 = st.columns([1,3])

    with st.spinner("Loading..."):
        with col1:
            ct1c1, ct1c2 = st.columns([1,2])
            with ct1c1:
                st.subheader("Price vs Volume Percentage Change")
            with ct1c2:
                st.write("")
                #create tickbox
                show_name = st.checkbox("🎉 Show Name")

    def VP_change(interval = "D", show_name = False):
        #setting percent change period
        period = None
        if interval == "D":
            period = 1
        elif interval == "W":
            period = 5
        elif interval == "M":
            period = 22
        elif interval == "3M":
            period = 65
        else:
            period = 1

        #check na
        select_df = dv[selection]
        check = select_df.iloc[-1].dropna().index.tolist()

        #price change
        pc = dp[check].pct_change(period).iloc[-1]
        pc = pc.T
        #volume
        pv = dv[check].ewm(span = 5, adjust = False).mean().pct_change(period).iloc[-1] #เรียก libs ไม่ได้
        pv = pv.T

        #combine
        cb_df = pd.DataFrame()
        cb_df["Symbols"] = pc.index
        cb_df[f"Price %Change {interval}"] =  pc.values*100
        cb_df[f"Volume %Change {interval}"] =  pv.values*100
        cb_df = cb_df.merge(rawt1, left_on= "Symbols" ,right_on= "symbol", how = "left")
        cb_df = cb_df.merge(mc, left_on = "Symbols", right_on = "Symbols", how = "left")

        df_use = cb_df.copy()
        data_source = df_use.drop(["symbol","full name", "market", "sector", "sub-sector"], axis = 1)

        #chart show
        text_interval = "Daily" if interval == "D" else "Weekly" if interval == "W" else "Montly" if interval == "M" else "Quaterly" if interval == "3M" else None
        VP_chart = px.scatter(df_use, x = f"Price %Change {interval}",
                                y = f"Volume %Change {interval}",
                                color = "sector",
                                text = "Symbols" if show_name else None,
                                custom_data= ["Symbols", "MarketCap", "sub-sector", f"Price %Change {interval}", f"Volume %Change {interval}","sector"],
                                size = "MarketCap",
                                title = f"{text_interval} Price vs Volume Percentage Change",
                                template = "seaborn")

        # update details in the hover
        VP_chart.update_traces(
        hovertemplate="<b>Name:</b> %{customdata[0]}<br>" +
                        "<b>Marketcap:</b> %{customdata[1]:,.0f} THB<extra></extra><br>" +
                        "<b>Sector:</b> %{customdata[5]}<br>"+
                        "<b>SubSector:</b> %{customdata[2]}<br>"+
                        f"<b>{text_interval} Price %Change: </b>" "%{customdata[3]:.2f}<br>"+
                        f"<b>{text_interval} Volume %Change: </b>" "%{customdata[4]:.2f}<br>",
        textposition = "bottom center",)

        VP_chart.update_layout(
            hovermode= "closest"
            )

        #add red line
        VP_chart.add_vline(x=0, line_color='red', line_width=2, opacity=0.6)
        VP_chart.add_hline(y=0, line_color='red', line_width=2, opacity=0.6)

        return VP_chart, data_source

    def RS_Score():
        #clean data by ffill then bfill
        data = dp.copy().ffill()
        data = data.bfill()

        #calcuate return quarterly
        Q1 = data.pct_change(65).iloc[-1]
        Q2 = data.pct_change(130).iloc[-1]
        Q3 = data.pct_change(195).iloc[-1]
        Q4 = data.pct_change(260).iloc[-1]

        #build dataframe
        rs_data = pd.DataFrame()
        rs_data["Q1"] = Q1
        rs_data["Q2"] = Q2
        rs_data["Q3"] = Q3
        rs_data["Q4"] = Q4
        rs_data["RS Score"] = np.round(0.4*Q1 + 0.2*Q2 + 0.2*Q3 + 0.2*Q4,4)

        #rs score
        rs_data["Rank"] = rs_data["RS Score"].rank(ascending= False, method = "max")
        rs_data.reset_index(inplace = True,)
        rs_data = rs_data.sort_values("Rank",ascending= True)
        rs_data["Return"] = rs_data["RS Score"]

        rs_data.rename(columns = {"index": "Name"}, inplace =True)
        rs_data["RS Score"] = 1 / (1+ np.e**(-rs_data["RS Score"]))

        #for display
        rs_data = rs_data.merge(rawt1, left_on = "Name", right_on = "symbol", how = "left")
        sorted_data = rs_data[["Name", "RS Score","market","sector", "sub-sector", "Rank", "Return"]] #"Q1", "Q2","Q3", "Q4"]]
        sorted_data.dropna(inplace = True)
        data_source = sorted_data.copy()
        # sorted_data.index = sorted_data.index.astype(int)
        sorted_data.set_index("Rank", inplace = True)
        sorted_data = sorted_data[sorted_data["Name"].isin(selection)]
        return sorted_data, data_source

    with col0:
        rs_score, rs_score_data = RS_Score()
        rs_score_data.attrs["description"] = "RS_Score Ranking contains a performance score"
        st.subheader("RS Ranking")
        st.dataframe(rs_score,height= 850, width= 500)

    with col1:
        scol0, scol1 = st.columns(2)
        with scol0:
            VP_change_D, VP_change_D_data = VP_change(interval = "D", show_name= show_name)
            st.plotly_chart(VP_change_D)
            VP_change_D_data.attrs["description"] = "Price and Volume Percentage Change in Daily"
        with scol1:
            VP_change_W, VP_change_W_data = VP_change(interval = "W", show_name= show_name)
            st.plotly_chart(VP_change_W)
            VP_change_W_data.attrs["description"] = "Price and Volume Percentage Change in Weekly"

        scol2, scol3 = st.columns(2)
        with scol2:
            VP_change_M, VP_change_M_data = VP_change(interval = "M", show_name= show_name)
            st.plotly_chart(VP_change_M)
            VP_change_M_data.attrs["description"] = "Price and Volume Percentage Change in Monthly"

        with scol3:
            VP_change_3M, VP_change_3M_data = VP_change(interval = "3M",  show_name= show_name)
            st.plotly_chart(VP_change_3M)
            VP_change_3M_data.attrs["description"] = "Price and Volume Percentage Change in Quaterly"
    st.divider()

# (The rest of the tab code continues here...)

st.divider()

#"""===================================AI Q&A Section==================================="""
# Create a tuple of all the dataframes that will be passed to the AI functions
# This is done once to avoid repeating the list
# Ensure all dataframes are created before this point
try:
    # This block assumes all dataframes for the tuple are generated inside the tabs
    # Re-running the VWAP function just to get the data for the AI
    def VWAP_for_AI(market = "SET", period_options = 90):
        temp_market = raw_std[raw_std["market"] == market]
        symbols_for_subplot = sorted(temp_market["symbol"].unique())
        price = dp.tail(period_options*2).copy().ffill()
        vol = dv.tail(period_options*2).copy().ffill()
        df_price = price * vol
        df_cum = df_price.cumsum()
        df_vol = vol.cumsum()
        df_VWAP = df_cum / df_vol
        df_VWAP = df_VWAP[symbols_for_subplot]
        df_VWAP_data = df_VWAP.T.reset_index().rename(columns={"index": "Name"})
        VWAP_cumprod = (df_VWAP.pct_change()+1).cumprod()
        VWAP_cumprod_data = VWAP_cumprod.T.reset_index().rename(columns={"index": "Name"})
        df_stack = dv.tail(period_options*2)[symbols_for_subplot].copy().ffill().bfill()
        df_VWAP_stack = df_VWAP.copy()
        df_stacked_bar = df_stack * df_VWAP_stack
        df_stacked_bar["sum"] = df_stacked_bar.sum(1)
        df_stacked_bar_ratio = df_stacked_bar.iloc[:, :-1].div(df_stacked_bar["sum"], axis=0)
        stacked_bar_ratio_datasource = df_stacked_bar_ratio.T
        return stacked_bar_ratio_datasource, df_VWAP_data, VWAP_cumprod_data

    SET_Stacked_bar, SET_VWAP, SET_VWAP_cumprod = VWAP_for_AI(market="SET")
    mai_Stacked_bar, mai_VWAP, mai_VWAP_cumprod = VWAP_for_AI(market="mai")

    # This part should be after all tabs have been processed to ensure data is available
    start = (datetime.datetime.now() - datetime.timedelta(100)).strftime("%Y-%m")
    geo_df = (dp[start:].copy().ffill().bfill().pct_change()+1).cumprod()
    geo_df = geo_df.reset_index().melt(id_vars=["Date"], var_name="Name", value_name="Return")
    geo_df = geo_df.merge(raw_std, left_on="Name", right_on="symbol", how="left")
    perf_source_SET = pd.pivot_table(geo_df[geo_df['market']=='SET'], values='Return', index='Name', columns='Date')
    perf_source_mai = pd.pivot_table(geo_df[geo_df['market']=='mai'], values='Return', index='Name', columns='Date')

    _52_df = dp.tail(265)
    _52_WHL_data = pd.DataFrame({
        'Name': _52_df.columns,
        'Close': _52_df.iloc[-1],
        'High': _52_df.max(),
        'Low': _52_df.min()
    }).merge(raw_std, left_on='Name', right_on='symbol')
    source_52SET = _52_WHL_data[_52_WHL_data['market'] == 'SET']
    source_52mai = _52_WHL_data[_52_WHL_data['market'] == 'mai']


    data_for_ai_tuple = (
        rs_score_data, VP_change_D_data, VP_change_W_data, VP_change_M_data, VP_change_3M_data,
        source_52SET, source_52mai, perf_source_SET, perf_source_mai,
        SET_Stacked_bar, SET_VWAP, SET_VWAP_cumprod,
        mai_Stacked_bar, mai_VWAP, mai_VWAP_cumprod
    )
except NameError as e:
    st.error(f"A dataframe required for the AI model was not created. Please ensure you have visited all tabs to generate the data. Error: {e}")
    # Create an empty tuple to prevent the app from crashing
    data_for_ai_tuple = (pd.DataFrame(),) * 15


# Create a placeholder for AI responses in the main area of the app
ai_response_placeholder = st.container()

with st.sidebar:
    st.header("🤖 AI Q&A")
    st.markdown("ถามคำถามเกี่ยวกับภาพรวมตลาดหรือหุ้นรายตัวจากข้อมูลที่แสดงผลในหน้านี้")

    user_question = st.text_area("ป้อนคำถามของคุณที่นี่...",
                                 value="หุ้นตัวไหนมี RS_Rank ดีที่สุด และมีแนวโน้มเป็นอย่างไร?",
                                 height=150,
                                 key="ai_question")

    if st.button("ส่งคำถามถึง AI", key="ask_ai_button"):
        if not user_question.strip():
            st.warning("กรุณาป้อนคำถาม")
        else:
            # This is the generic prompt for the AI
            generic_prompt = f"""
                **Persona:** You are a highly skilled Senior Financial Analyst. Your sole purpose is to answer the user's question based *strictly and exclusively* on the technical data provided in the attached CSV file. Do not use any external knowledge, news, or fundamental data. Your credibility depends on analyzing only the data given.

                **User's Question:** "{user_question}"

                **Core Task:**
                1.  Thoroughly analyze the provided CSV data to find the information relevant to the user's question.
                2.  Formulate a concise, data-driven answer.
                3.  **Crucially, you must back up every claim with specific data points from the file.** For example, instead of saying "The stock is strong," say "The stock shows strong momentum, evidenced by its RS_Rank of X and a positive 'Price %Change D' of Y%."
                4.  If the data required to answer the question is missing or contains 'NaN', you must explicitly state that the specific data is unavailable and then provide the best possible answer with the remaining data.

                **Data Dictionary (for your reference):**
                * **RS_Rank:** Momentum ranking vs. the market (lower is better).
                * **Price/Volume %Change (D, W, M, 3M):** Trend acceleration/deceleration.
                * **52WHL:** Position in 52-week range. >0.8 is near highs, <0.2 is near lows.
                * **ROC_10:** Short-term (2-week) momentum.
                * **Performance 100 Days:** Mid-term cumulative performance.
                * **Volume Trade Ratio in Its Sector:** Market share of trading volume.
                * **Cumulative_VWAP:** Cumulative Volume-Weighted Average Price trend.

                **Language:** Your response must be in Thai (ภาษาไทย).
            """
            # Call the generic AI function and get the response text
            response_text = get_ai_response(generic_prompt, data_for_ai_tuple)

            # Display the response in the placeholder on the main page
            if response_text:
                with ai_response_placeholder:
                    st.markdown("### 🤖 คำตอบจาก AI")
                    st.markdown("---")
                    st.markdown(response_text)
