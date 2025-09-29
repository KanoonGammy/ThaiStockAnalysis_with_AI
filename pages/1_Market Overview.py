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
    def VP_change(interval, show_name, selection_list):
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
    def RS_Score(selection_list):
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
        geo_fig = px.line(data, x = "Date", y = "Return", color = "Name",
                          title = f"Performance: Sector:<br> &nbsp;Sector: <i>{select_sector}<i> <br> &nbsp;Sub-Sector: <i>{select_sub or 'ALL'}</i>",
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
        fig_52WHL = px.scatter(data, x = "ROC_10", y = "52WHL", color = "Name", size="MarketCap",
                               text="Name",
                               custom_data=["Name", "ROC_10", "52WHL", "sector", "sub-sector", "MarketCap", 'Close'],
                               title = f"52 Weeks High Low vs Rate of Change 10:<br> &nbsp;Sector: <i>{select_sector}<i> <br> &nbsp;Sub-Sector: <i>{select_sub or 'ALL'}</i>",
                               template = "seaborn")
        
        # FIX 2: Add red lines back
        fig_52WHL.add_vline(x=0, line_color='red', line_width=2, opacity=0.6)
        fig_52WHL.add_hline(y=0, line_color='red', line_width=2, opacity=0.6)
        fig_52WHL.add_hline(y=1, line_color='red', line_width=2, opacity=0.6)

        fig_52WHL.update_traces(
            hovertemplate="<b>Name:</b> %{customdata[0]}<br><b>Close:</b> %{customdata[6]:,.2f} <br><b>ROC_10:</b> %{customdata[1]:,.2f}% <br><b>52WHL:</b> %{customdata[2]:,.2f} <br><b>sector:</b> %{customdata[3]} <br><b>sub-sector:</b> %{customdata[4]} <br><b>MarketCap:</b> %{customdata[5]:,.0f} THB <br>",
            textposition = "bottom center"
        )
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

#"""===================================TAB 3 (RESTORED) ==================================="""
with tab3:
    tab31, tab32 = st.tabs(["\t\tSET\t\t", "\t\tmai\t\t"])

    def Volume_Analysis_iteration_by_Sector(market, data_volume_in_wide_format, period, fig_width=600, fig_height=1200):
        temp_market = raw_std[raw_std["market"] == market]
        sectors_for_subplot = sorted(temp_market["sector"].unique())
        fig_VA = make_subplots(rows = len(sectors_for_subplot), cols = 1, subplot_titles= sectors_for_subplot)
        for i,j in enumerate(sectors_for_subplot):
            temp_sector = temp_market[temp_market["sector"]== j]
            temp_tickers = sorted(temp_sector["symbol"].unique())
            temp_mc = mc[mc["Symbols"].isin(temp_tickers)].copy()
            temp_mc["weight"] = temp_mc["MarketCap"] / temp_mc["MarketCap"].sum()
            VA = data_volume_in_wide_format.tail(period + 70).copy().fillna(0)
            VA = VA[temp_tickers].reset_index()
            VA = pd.melt(VA, id_vars = ["Date"], var_name= "Name", value_name= "Volume")
            VA = VA.merge(right = temp_mc, left_on= "Name", right_on = "Symbols").drop("Symbols", axis =1)
            VA["Weighted_Volume"] = VA["Volume"] * VA["weight"]
            VA = pd.pivot_table(VA, index = "Date", columns = "Name", values = "Weighted_Volume")
            VA["Sum"] = VA.sum(1)
            VA["Threshold"] = VA["Sum"].rolling(window = 22).mean() + 2 * VA["Sum"].rolling(window = 22).std()
            VA["Anomaly"] = VA["Sum"] > VA["Threshold"]
            VA = VA.reset_index().tail(period)
            fig_VA.add_trace(go.Bar(x = VA["Date"], y = VA["Sum"], name = j,showlegend= False, marker = dict(color = VA["Anomaly"].map({True: "orange", False:"gray"}))), row= i+1,col=1)
        fig_VA.update_layout(width = fig_width, height = fig_height, title = f"{market} Sector Volume Analysis Period: {period} Days", template="seaborn")
        return fig_VA

    def Volume_Analysis_iteration_by_Stock(market, sector, data_volume_in_wide_format, period, page, fig_width=600, fig_height=1200):
        temp_market = raw_std[raw_std["market"] == market]
        temp_sector = temp_market[temp_market["sector"] == sector]
        symbols_for_subplot = sorted(temp_sector["symbol"].unique())[10*(page - 1) :page * 10]
        if not symbols_for_subplot:
            st.warning("ไม่มีหุ้นให้แสดงในหน้านี้")
            return go.Figure()
        fig_symbols = make_subplots(rows = len(symbols_for_subplot), cols = 1, subplot_titles= symbols_for_subplot)
        for i,j in enumerate(symbols_for_subplot):
            Vol_df = data_volume_in_wide_format[symbols_for_subplot].tail( period + 70 ).copy().fillna(0)
            Vol_threshold = Vol_df.rolling(window = 22).mean() + 2 * Vol_df.rolling(window = 22).std()
            Vol_anomaly = Vol_df > Vol_threshold
            cutoff_for_plot = pd.DataFrame({
                "Date": Vol_df.index,
                "Volume": Vol_df[j],
                "Anomaly": Vol_anomaly[j]
            }).dropna().tail(period)
            fig_symbols.add_trace(go.Bar(x = cutoff_for_plot["Date"], y = cutoff_for_plot["Volume"], name = j, showlegend= False, marker = dict(color = cutoff_for_plot["Anomaly"].map({True: "orange", False:"gray"}))), row= i+1, col=1)
        fig_symbols.update_layout(width = fig_width, height = fig_height, title = f"{market} Sector: {sector} Volume Analysis Period: {period} Days ", template="seaborn")
        return fig_symbols

    def VWAP(market, sector, period, options:str = None):
        temp_market = raw_std[raw_std["market"] == market]
        temp_sector = temp_market[temp_market["sector"] == sector] if sector is not None else temp_market
        symbols_for_subplot = sorted(temp_sector["symbol"].unique())
        if not symbols_for_subplot:
            return go.Figure(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        price = dp[symbols_for_subplot].tail(period*2).copy().ffill().bfill()
        vol = dv[symbols_for_subplot].tail(period*2).copy().ffill().bfill()
        df_VWAP = (price * vol).cumsum() / vol.cumsum()
        df_VWAP_data = df_VWAP.T.reset_index().rename(columns = {"index": "Name"})
        VWAP_cumprod = (df_VWAP.pct_change()+1).cumprod()
        VWAP_cumprod_data = VWAP_cumprod.T.reset_index().rename(columns = {"index": "Name"})
        df_stacked_bar = (vol * df_VWAP)
        df_stacked_bar["sum"] = df_stacked_bar.sum(1)
        df_stacked_bar_ratio = df_stacked_bar.iloc[:, :-1].div(df_stacked_bar["sum"], axis = 0).reset_index()
        stacked_bar_ratio_datasource = df_stacked_bar_ratio.set_index("Date").T
        fig = make_subplots(rows = 2, cols = 1, subplot_titles= ["VWAP Cumulative Product", "Stacked Volume Ratio"], shared_xaxes=True)
        for column in VWAP_cumprod.columns:
            fig.add_trace(go.Scatter(x=VWAP_cumprod.index, y=VWAP_cumprod[column], mode="lines", name=column, legendgroup='1'), row=1, col=1)
            fig.add_trace(go.Bar(x=df_stacked_bar_ratio["Date"], y=df_stacked_bar_ratio[column], name=column, legendgroup='2', showlegend=False), row=2, col=1)
        fig.update_layout(title_text=f"VWAP Analysis: {market} - {sector}", template="seaborn", height=800, barmode="stack")
        return fig, stacked_bar_ratio_datasource, df_VWAP_data, VWAP_cumprod_data

    with tab31:
        t31c1, t31c2, t31c3 = st.columns([2,3,2])
        with t31c1:
            st.subheader("SET Sector Volume")
            p_set_sec = st.radio("Period", [30,60,90], key="p_set_sec", horizontal=True)
            st.plotly_chart(Volume_Analysis_iteration_by_Sector("SET", dv, p_set_sec, fig_height=900), use_container_width=True)
        with t31c2:
            st.subheader("SET VWAP Analysis")
            s_set_vwap = st.radio("Select Sector", sorted(raw_std[raw_std["market"] == "SET"]['sector'].unique()), key="s_set_vwap", horizontal=True)
            p_set_vwap = st.radio("Period", [30,60,90], key="p_set_vwap", horizontal=True)
            fig_set, set_stack, set_vwap, set_cumprod = VWAP("SET", s_set_vwap, p_set_vwap)
            st.plotly_chart(fig_set, use_container_width=True)
        with t31c3:
            st.subheader("SET Stock Volume")
            s_set_stk = st.radio("Select Sector", sorted(raw_std[raw_std["market"] == "SET"]['sector'].unique()), key="s_set_stk", horizontal=True)
            p_set_stk = st.radio("Period", [30,60,90], key="p_set_stk", horizontal=True)
            page_max_set = (len(raw_std[(raw_std["market"] == "SET") & (raw_std["sector"] == s_set_stk)]) // 10) + 1
            pg_set_stk = st.radio("Page", range(1, page_max_set + 1), key="pg_set_stk", horizontal=True)
            st.plotly_chart(Volume_Analysis_iteration_by_Stock("SET", s_set_stk, dv, p_set_stk, pg_set_stk, fig_height=900), use_container_width=True)

    with tab32:
        t32c1, t32c2, t32c3 = st.columns([2,3,2])
        with t32c1:
            st.subheader("mai Sector Volume")
            p_mai_sec = st.radio("Period", [30,60,90], key="p_mai_sec", horizontal=True)
            st.plotly_chart(Volume_Analysis_iteration_by_Sector("mai", dv, p_mai_sec, fig_height=900), use_container_width=True)
        with t32c2:
            st.subheader("mai VWAP Analysis")
            s_mai_vwap = st.radio("Select Sector", sorted(raw_std[raw_std["market"] == "mai"]['sector'].unique()), key="s_mai_vwap", horizontal=True)
            p_mai_vwap = st.radio("Period", [30,60,90], key="p_mai_vwap", horizontal=True)
            fig_mai, mai_stack, mai_vwap, mai_cumprod = VWAP("mai", s_mai_vwap, p_mai_vwap)
            st.plotly_chart(fig_mai, use_container_width=True)
        with t32c3:
            st.subheader("mai Stock Volume")
            s_mai_stk = st.radio("Select Sector", sorted(raw_std[raw_std["market"] == "mai"]['sector'].unique()), key="s_mai_stk", horizontal=True)
            p_mai_stk = st.radio("Period", [30,60,90], key="p_mai_stk", horizontal=True)
            page_max_mai = (len(raw_std[(raw_std["market"] == "mai") & (raw_std["sector"] == s_mai_stk)]) // 10) + 1
            pg_mai_stk = st.radio("Page", range(1, page_max_mai + 1), key="pg_mai_stk", horizontal=True)
            st.plotly_chart(Volume_Analysis_iteration_by_Stock("mai", s_mai_stk, dv, p_mai_stk, pg_mai_stk, fig_height=900), use_container_width=True)
    
    # For AI data gathering
    _, SET_Stacked_bar, SET_VWAP, SET_VWAP_cumprod = VWAP(market = "SET", sector = None, period=90, options= "datasource")
    _, mai_Stacked_bar, mai_VWAP, mai_VWAP_cumprod= VWAP(market = "mai", sector = None, period=90, options= "datasource")


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
            hm_df["M"] = hm_df.Date.dt.month
            pivoted_hm_df = pd.pivot_table(hm_df, index = "Y", columns = "M", values = "Return")
            pivoted_hm_df.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            # FIX 4: Change font size with annot_kws
            sns.heatmap(pivoted_hm_df, cmap="RdYlGn", annot=True, fmt=".2%", ax=ax, center=0, annot_kws={"size": 8})
            plt.title(f"Historical Monthly Return Heatmap of {hm_name}")
            plt.ylabel("Year")
            plt.xlabel("Month")
            st.pyplot(fig)
        else:
            if hm_name: st.error(f"Stock '{hm_name}' not found.")

#"""===================================AI SECTION==================================="""
# The main app area no longer needs a placeholder
# with st.sidebar: is now the single place for AI interaction

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
            
            # FIX 1: Display response inside the sidebar
            st.markdown("---")
            st.markdown("### 🤖 คำตอบจาก AI")
            st.markdown(response_text)

