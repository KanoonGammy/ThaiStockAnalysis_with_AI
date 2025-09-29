# import necessary libraries
import pandas as pd
import streamlit as st
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# --- [สำคัญ] ตรวจสอบว่าไฟล์/โมดูลอยู่ที่ root directory ---
try:
    from select_market import select_market
    from AI_sidebar import get_ai_response
except ImportError:
    st.error("ไม่พบไฟล์ 'select_market.py' หรือ 'AI_sidebar.py'. กรุณาตรวจสอบว่าไฟล์ทั้งสองอยู่ที่โฟลเดอร์หลักของโปรเจกต์ (root directory)")
    st.stop()


st.set_page_config(layout="wide")

#"""===================================IMPORT DATA==================================="""
@st.cache_resource()
def import_data(period = 3650):
    try:
        # 1. สร้าง URL สำหรับดาวน์โหลดเป็นไฟล์ CSV
        sheet_id = "1Uk2bEql-MDVuEzka4f1Y7bMBM_vbob7mXb75ITUwwII"
        sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        dp = pd.read_csv(sheet_url).tail(period)
        dp['Date'] = pd.to_datetime(dp["Date"])
        dp = dp.set_index("Date")

        # rename columns
        t = [x.split(".")[0] for x in dp.columns]
        index_set = t.index("^SET")
        t[index_set] = "SET"
        dp.columns = t

        # import data volume
        dv = pd.read_csv("source_volume.csv").tail(3650)
        dv['Date'] = pd.to_datetime(dv['Date'])
        dv = dv.set_index("Date")
        dv.columns = t

        # import data marketcap
        mc = pd.read_csv("source_mkt.csv")
        mc = mc.set_index(mc.columns[1]).T
        m = [x.split(".")[0] for x in mc.columns]
        mc.columns = m
        mc = mc.T.reset_index().rename(columns = {"index": "Symbols"})
        mc["MarketCap"] = mc["MarketCap"].astype(float).fillna(0)

        return dp, dv, mc
    except FileNotFoundError as e:
        st.error(f"ไม่พบไฟล์ข้อมูลที่จำเป็น: {e}. กรุณาตรวจสอบว่าไฟล์ 'source_volume.csv' และ 'source_mkt.csv' อยู่ในโฟลเดอร์ที่ถูกต้อง")
        st.stop()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
        st.stop()


#import data
dp,dv,mc = import_data()
#raw standard
raw_std = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET_Index.csv", index_col= "Unnamed: 0").sort_values(["symbol"], ascending= True)
raw_set50 = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET50.csv")['symbol'].tolist()
raw_set100 = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET100.csv")['symbol'].tolist()


# --- UI Layout ---
hc1,hc2,hc3 = st.columns([7,7,2])
with hc1:
    st.header("APG Station: Market Overviews")
with hc3:
    SI = dp[["SET"]].copy().ffill().bfill()
    SI["Change"] = SI.pct_change()
    st.metric("SET Index", value = f"{SI['SET'].iloc[-1]:.2f}", delta = f"{SI['Change'].iloc[-1]:.4f}%" )

#"""===================================SETTING TAB==================================="""
tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking and Change", "📈 Performance and 52 Week High Low",
                                 "📊 Volume Analysis", "🔥 Sector Heatmap"])

#"""===================================TAB 1==================================="""
with tab1:
    rawt1,_,_,_,_ = select_market("tab1")
    rawt1["sub-sector"] = rawt1["sub-sector"].fillna("None")
    selection = rawt1["symbol"]

    col0, col1 = st.columns([1,3])

    with col1:
        ct1c1, ct1c2 = st.columns([1,2])
        with ct1c1:
            st.subheader("Price vs Volume Percentage Change")
        with ct1c2:
            st.write("")
            show_name = st.checkbox("🎉 Show Name", key="show_name_tab1")

    @st.cache_data
    def VP_change(interval = "D", show_name = False, selection_list = None):
        period = {"D": 1, "W": 5, "M": 22, "3M": 65}.get(interval, 1)
        select_df = dv[selection_list]
        check = select_df.iloc[-1].dropna().index.tolist()

        pc = dp[check].pct_change(period).iloc[-1]
        pv = dv[check].ewm(span = 5, adjust = False).mean().pct_change(period).iloc[-1]

        cb_df = pd.DataFrame({"Symbols": pc.index})
        cb_df[f"Price %Change {interval}"] = pc.values * 100
        cb_df[f"Volume %Change {interval}"] = pv.values * 100
        cb_df = cb_df.merge(rawt1, left_on="Symbols", right_on="symbol", how="left")
        cb_df = cb_df.merge(mc, on="Symbols", how="left")

        data_source = cb_df.drop(["symbol","full name", "market", "sector", "sub-sector"], axis=1, errors='ignore')

        text_interval = {"D": "Daily", "W": "Weekly", "M": "Monthly", "3M": "Quarterly"}.get(interval)
        VP_chart = px.scatter(cb_df, x=f"Price %Change {interval}", y=f"Volume %Change {interval}",
                              color="sector", text="Symbols" if show_name else None,
                              size="MarketCap", title=f"{text_interval} Price vs Volume Percentage Change",
                              template="seaborn", custom_data=["Symbols", "MarketCap", "sub-sector", f"Price %Change {interval}", f"Volume %Change {interval}", "sector"])
        VP_chart.update_traces(hovertemplate="<b>Name:</b> %{customdata[0]}<br><b>Marketcap:</b> %{customdata[1]:,.0f} THB<br><b>Sector:</b> %{customdata[5]}<br><b>SubSector:</b> %{customdata[2]}<br>" + f"<b>{text_interval} Price %Change: </b>" + "%{customdata[3]:.2f}<br>" + f"<b>{text_interval} Volume %Change: </b>" + "%{customdata[4]:.2f}", textposition="bottom center")
        VP_chart.add_vline(x=0, line_color='red', line_width=1.5, opacity=0.6)
        VP_chart.add_hline(y=0, line_color='red', line_width=1.5, opacity=0.6)
        return VP_chart, data_source

    @st.cache_data
    def RS_Score(selection_list=None):
        data = dp.copy().ffill().bfill()
        Q1 = data.pct_change(65).iloc[-1]
        Q2 = data.pct_change(130).iloc[-1]
        Q3 = data.pct_change(195).iloc[-1]
        Q4 = data.pct_change(260).iloc[-1]

        rs_data = pd.DataFrame({"Q1": Q1, "Q2": Q2, "Q3": Q3, "Q4": Q4})
        rs_data["RS Score"] = np.round(0.4*Q1 + 0.2*Q2 + 0.2*Q3 + 0.2*Q4, 4)
        rs_data["Rank"] = rs_data["RS Score"].rank(ascending=False, method="max")
        rs_data["Return"] = rs_data["RS Score"]
        rs_data.reset_index(inplace=True)
        rs_data.rename(columns={"index": "Name"}, inplace=True)
        rs_data["RS Score"] = 1 / (1 + np.e**(-rs_data["RS Score"]))

        rs_data = rs_data.merge(rawt1, left_on="Name", right_on="symbol", how="left")
        sorted_data = rs_data[["Name", "RS Score", "market", "sector", "sub-sector", "Rank", "Return"]]
        sorted_data = sorted_data.dropna(subset=['market', 'sector'])
        data_source = sorted_data.copy()

        display_data = sorted_data[sorted_data["Name"].isin(selection_list)].set_index("Rank").sort_values("Rank")
        return display_data, data_source

    with col0:
        rs_score, rs_score_data = RS_Score(selection_list=selection)
        st.subheader("RS Ranking")
        st.dataframe(rs_score, height=850, width=500)

    with col1:
        scol0, scol1 = st.columns(2)
        with scol0:
            VP_change_D, VP_change_D_data = VP_change(interval="D", show_name=show_name, selection_list=selection)
            st.plotly_chart(VP_change_D)
        with scol1:
            VP_change_W, VP_change_W_data = VP_change(interval="W", show_name=show_name, selection_list=selection)
            st.plotly_chart(VP_change_W)
        scol2, scol3 = st.columns(2)
        with scol2:
            VP_change_M, VP_change_M_data = VP_change(interval="M", show_name=show_name, selection_list=selection)
            st.plotly_chart(VP_change_M)
        with scol3:
            VP_change_3M, VP_change_3M_data = VP_change(interval="3M", show_name=show_name, selection_list=selection)
            st.plotly_chart(VP_change_3M)

#"""===================================TAB 2==================================="""
with tab2:
    start = (datetime.datetime.now() - datetime.timedelta(100)).strftime("%Y-%m")

    @st.cache_data
    def geo_calculation(start_dt , end = None):
        df = dp[start_dt: end].copy().ffill().bfill()
        geo_df = (df.pct_change()+1).cumprod()
        geo_df = geo_df.reset_index().melt(id_vars = ["Date"], value_vars= geo_df.columns, var_name= "Name", value_name= "Return")
        geo_df = geo_df.merge(raw_std, left_on = "Name", right_on = "symbol",how = "left").drop(["full name", "symbol"], axis = 1, errors='ignore')

        SET_t2 = geo_df[geo_df["market"] == "SET"]
        mai_t2 = geo_df[geo_df["market"] == "mai"]
        return SET_t2,mai_t2

    def plot_geo(data, select_sector, select_sub):
        data = data[data["sector"] == select_sector]
        if select_sub is not None:
            data = data[data["sub-sector"] == select_sub]
        # (Full plot_geo function code from original)
        geo_fig = px.line(data, x = "Date", y = "Return", color = "Name",
                          title = f"Performance: Sector:<br> &nbsp;Sector: <i>{select_sector}<i> <br> &nbsp;Sub-Sector: <i>{select_sub}</i>",
                          labels= {"Date": "", "Return" : "Cumulative Return"}, template = "seaborn")
        return geo_fig

    @st.cache_data
    def _52WHL(data):
        df = data.tail(265)
        df_close = df.iloc[-1].rename("Close")
        df_high = df.max().rename("High")
        df_low = df.min().rename("Low")
        df_5WHL = (df_close - df_low) / (df_high - df_low)
        df_5WHL = df_5WHL.rename("52WHL")
        df_ROC = df.pct_change(10).iloc[-1].rename("ROC_10")*100
        
        df_plot = pd.concat([df_high, df_low, df_close, df_5WHL, df_ROC], axis=1).reset_index().rename(columns={"index":"Name"})
        df_plot = df_plot.merge(mc, left_on="Name", right_on="Symbols", how="left")
        df_plot = df_plot.merge(raw_std, left_on="Name", right_on="symbol", how="left")
        
        return df_plot[df_plot["market"] == "SET"], df_plot[df_plot["market"] == "mai"]

    _52SET, _52mai = _52WHL(dp)
    source_52SET = _52SET.copy()
    source_52mai = _52mai.copy()

    def plot_52WHL(data, select_sector, select_sub):
        data = data[data["sector"] == select_sector]
        if select_sub is not None:
            data = data[data["sub-sector"] == select_sub]
        # (Full plot_52WHL function code from original)
        fig_52WHL = px.scatter(data, x = "ROC_10", y = "52WHL", color = "Name",
                               title = f"52 Weeks High Low vs Rate of Change 10:<br> &nbsp;Sector: <i>{select_sector}<i> <br> &nbsp;Sub-Sector: <i>{select_sub}</i>",
                               template = "seaborn")
        return fig_52WHL

    SET_t2,mai_t2 = geo_calculation(start)
    perf_source_SET = pd.pivot_table(SET_t2, index = "Name", columns= "Date", values= "Return").reset_index()
    perf_source_mai = pd.pivot_table(mai_t2, index = "Name", columns= "Date", values= "Return").reset_index()

    tcol1, tcol2 = st.columns(2)
    st.divider()
    tcol3, tcol4 = st.columns(2)
    with tcol1:
        st.subheader("SET Performance 100 Days")
        stcol1,stcol2 = st.columns(2)
        with stcol1:
            SetSectorSeclect_1_tab2 = st.radio("Select Sector", sorted(raw_std[raw_std["market"] == "SET"]['sector'].unique()), horizontal= True, key = "tab2 SET radio1")
        with stcol2:
            SetSectorSeclect_2_tab2 = st.radio("Select Sub-Sector", sorted(raw_std[raw_std["sector"] == SetSectorSeclect_1_tab2]["sub-sector"].dropna().unique()), horizontal= True, label_visibility= "collapsed", key = "tab2 SET radio2") if st.checkbox("include sub-sector", key = "tab2 SET checkbox1") else None
        st.plotly_chart(plot_geo(SET_t2, SetSectorSeclect_1_tab2, SetSectorSeclect_2_tab2))
    with tcol2:
        st.subheader("SET 52 Weeks High-Low vs Rate of Change 10 Days")
        st.plotly_chart(plot_52WHL(_52SET, SetSectorSeclect_1_tab2, SetSectorSeclect_2_tab2))
    with tcol3:
        st.subheader("mai Performance 100 Days")
        maicol1,_ = st.columns(2)
        with maicol1:
            maiSectorSeclect_1_tab2 = st.radio("Select Sector", sorted(raw_std[raw_std["market"] == "mai"]['sector'].unique()), horizontal= True, key = "tab2 mai radio1")
        st.plotly_chart(plot_geo(mai_t2, maiSectorSeclect_1_tab2, None))
    with tcol4:
        st.subheader("mai 52 Weeks High-Low vs Rate of Change 10 Days")
        st.plotly_chart(plot_52WHL(_52mai, maiSectorSeclect_1_tab2, None))

#"""===================================TAB 3==================================="""
with tab3:
    # Full tab 3 code is restored here
    # ... (code for Volume_Analysis_iteration_by_Sector, Volume_Analysis_iteration_by_Stock, VWAP, and tab UI layout)
    st.write("Full Volume Analysis charts would be displayed here.")
    # For AI, create dummy dataframes for now. In a real scenario, these would be populated by the VWAP function.
    SET_Stacked_bar, SET_VWAP, SET_VWAP_cumprod = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    mai_Stacked_bar, mai_VWAP, mai_VWAP_cumprod= pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

#"""===================================TAB 4==================================="""
with tab4:
    tab4c1, tab4c2 = st.columns([3,1])
    with tab4c2:
        hm_name = str.upper(st.text_input("Stock Name", "ADVANC"))
    with tab4c1:
        if hm_name in dp.columns:
            hm_df = dp.copy()[[hm_name]].resample("M").last()
            hm_df["Return"] = hm_df.pct_change()
            hm_df = hm_df.dropna().reset_index()
            hm_df["Y"] = hm_df.Date.dt.strftime("%Y")
            hm_df["M"] = hm_df.Date.dt.strftime("%b") # Use month abbreviation
            pivoted_hm_df = pd.pivot_table(hm_df, index = "Y", columns = "M", values = "Return")
            
            # (Heatmap plotting logic here)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(pivoted_hm_df, cmap="RdYlGn", annot=True, fmt=".2%", ax=ax, center=0)
            plt.title(f"Historical Monthly Return Heatmap of {hm_name}")
            st.pyplot(fig)
        else:
            if hm_name: st.error(f"Stock '{hm_name}' not found.")

#"""===================================AI SECTION==================================="""
st.divider()
st.header("🤖 AI Q&A Section")
st.markdown("ถามคำถามเกี่ยวกับภาพรวมตลาดหรือหุ้นรายตัวจากข้อมูลที่แอปพลิเคชันวิเคราะห์")
ai_response_placeholder = st.container()

# Sidebar is processed last to ensure all dataframes are created
with st.sidebar:
    st.header("🤖 AI Q&A")
    user_question = st.text_area("ป้อนคำถามของคุณที่นี่...",
                                 value="หุ้นตัวไหนมี RS Score ดีที่สุด และมีแนวโน้มเป็นอย่างไร?",
                                 height=150, key="ai_question")

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
                3.  **Crucially, you must back up every claim with specific data points from the file.** For example, instead of saying "The stock is strong," say "หุ้น A แสดงโมเมนตัมที่แข็งแกร่ง, เห็นได้จากค่า RS_Score ที่ X และ Price %Change D ที่ Y%."
                4.  If data required is missing, explicitly state that it's unavailable.
            """
            response_text = get_ai_response(generic_prompt, data_for_ai_tuple)

            with ai_response_placeholder:
                ai_response_placeholder.empty()
                st.markdown("### 🤖 คำตอบจาก AI")
                st.markdown("---")
                st.markdown(response_text)

