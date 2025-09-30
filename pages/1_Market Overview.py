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
from AI_sidebar import get_ai_response # Import the updated function

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
    dv = pd.read_csv("source_volume.csv",).tail(3650)
    dv.Date = pd.to_datetime(dv['Date'])
    dv = dv.set_index("Date")
    dv.columns = t #use columns dp

    # import data marketcap
    mc = pd.read_csv("source_mkt.csv")
    mc = mc.set_index(mc.columns[1]).T
    m = [x.split(".")[0] for x in mc.columns]
    mc.columns = m
    mc = mc.T.reset_index().rename(columns = {"index": "Symbols"})
    mc["MarketCap"] = mc["MarketCap"].astype(float).fillna(0)

    return dp,dv,mc

#import data
dp,dv,mc = import_data()
#raw standard
raw_std = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET_Index.csv", index_col= "Unnamed: 0").sort_values(["symbol"], ascending= True)
raw_set50 = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET50.csv")['symbol'].tolist()
raw_set100 = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET100.csv")['symbol'].tolist()

hc1,hc2,hc3, hc4 = st.columns([6,6,2,4])
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

with hc4:
    last_update_date = dp.index[-1].strftime('%d %b %Y')
    st.metric("Last Data Update", value=last_update_date)

st.divider()

#"""===================================GLOBAL FILTERS==================================="""
rawt1,sl11,sl12,sl13,sl14 = select_market("global_filters")
rawt1["sub-sector"] = rawt1["sub-sector"].fillna("None")

selection = rawt1.reset_index(drop = True)
selection.index = selection.index + 1
selection = selection["symbol"]

st.divider()

#"""===================================SETTING TAB==================================="""
tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking and Change", "📈 Performance and 52 Week High Low",
                                 "📊 Volume Analysis", "🔥 Sector Heatmap"])

#"""===================================TAB 1==================================="""
with tab1:
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
        # st.write("rs_score_data") ################################################ data source ################################################

    with col1:
        scol0, scol1 = st.columns(2)

        with scol0:

            VP_change_D, VP_change_D_data = VP_change(interval = "D", show_name= show_name)
            # st.write("VP_change_D_data") ################################################ data source ################################################
            st.plotly_chart(VP_change_D)
            VP_change_D_data.attrs["description"] = "Price and Volume Percentage Change in Daily"

        with scol1:
            VP_change_W, VP_change_W_data = VP_change(interval = "W", show_name= show_name)
            # st.write("VP_change_W_data") ################################################ data source ################################################
            st.plotly_chart(VP_change_W)
            VP_change_W_data.attrs["description"] = "Price and Volume Percentage Change in Weekly"


        scol2, scol3 = st.columns(2)
        with scol2:
            VP_change_M, VP_change_M_data = VP_change(interval = "M", show_name= show_name)
            # st.write("VP_change_M_data") ################################################ data source ################################################
            st.plotly_chart(VP_change_M)
            VP_change_M_data.attrs["description"] = "Price and Volume Percentage Change in Monthly"


        with scol3:
            VP_change_3M, VP_change_3M_data = VP_change(interval = "3M",  show_name= show_name)
            # st.write("VP_change_3M_data") ################################################ data source ################################################
            st.plotly_chart(VP_change_3M)
            VP_change_3M_data.attrs["description"] = "Price and Volume Percentage Change in Quaterly"

#"""===================================TAB 2==================================="""
with tab2:
    st.markdown("""
    <style>
    div.stRadio {
        margin-top: 0.1px;  /* ระยะห่างด้านบน */
        margin-bottom: 0.1px;  /* ระยะห่างด้านล่าง */
        gap: 0.2px;
    }
    div.stCheckbox {
    margin-top: 0.1px;  /* ลดระยะห่างด้านบนของ checkbox */
    margin-bottom: 0.1px;  /* ลดระยะห่างด้านบนของ checkbox */
    }
    </style>
    """, unsafe_allow_html=True)
    start = (datetime.datetime.now() - datetime.timedelta(100)).strftime("%Y-%m")

    #=================== Performance 100 days =========================
    @st.cache_resource
    def geo_calculation(start , end = None):
        df = dp[start: end].copy().ffill().bfill()
        geo_df = (df.pct_change()+1).cumprod()
        geo_df = geo_df.reset_index().melt(id_vars = ["Date"], value_vars= geo_df.columns, var_name= "Name", value_name= "Return")
        geo_df = geo_df.merge(raw_std, left_on = "Name", right_on = "symbol",how = "left").drop(["full name", "symbol"], axis = 1)

        SET_t2 = geo_df[geo_df["market"] == "SET"]
        mai_t2 = geo_df[geo_df["market"] == "mai"]

        return SET_t2,mai_t2

    # @st.cache_resource
    def plot_geo(data, select_sector, select_sub):
        data = data[data["sector"] == select_sector]
        if select_sub != None:
            data = data[data["sub-sector"] == select_sub]
        geo_fig = px.line(data, x = "Date",
                          y = "Return",
                          color = "Name",
                          title = f"Performance: Sector:<br> &nbsp;Sector: <i>{select_sector}<i> <br> &nbsp;Sub-Sector: <i>{select_sub}</i>",
                          labels= {"Date": "", "Return" : "Cumulative Return"}, template = "seaborn"
                          )
        geo_fig.update_xaxes(rangeslider_visible=True)
        geo_fig.update_layout(
            xaxis = dict(
                rangeselector = dict(
                    buttons = [
                        dict(count = 7, label = "1w", step = "day", stepmode = "backward"),
                        dict(count = 1, label = "1m", step = "month", stepmode = "backward"),
                        dict(count = 3, label = "3m", step = "month", stepmode = "backward"),
                        # dict(count = 6, label = "6m", step = "month", stepmode = "backward"),
                        # dict(count = 1, label = "1y", step = "year", stepmode = "backward"),
                        # dict(count = 3, label = "3y", step = "year", stepmode = "backward"),
                        # dict(label = "YTD", step = "year", stepmode = "todate")
                    ], xanchor = "left", yanchor = "top"
                ),
            rangeslider = dict(visible = False),
            type = "date"
            ),
            yaxis=dict(
                autorange=True,         # ให้แกน Y ปรับช่วงอัตโนมัติ
                fixedrange=False,       # ปลดล็อกการเลื่อน/ขยายแกน Y
                showspikes=True,        # เพิ่มเส้น spike ชี้จุดข้อมูล
                spikemode='across'      # แสดง spike ทั้งแกน X และ Y
            ),
            dragmode="zoom"
        )
        # geo_fig.update_yaxes(dtick=1)

        return geo_fig

    #=================== 52Week High Low =========================
    def _52WHL(data):
        df = data.tail(265)
        df_close = df.iloc[-1].rename("Close")
        df_high = df.max().rename("High")
        df_low = df.min().rename("Low")
        df_5WHL = (df - df_low) / (df_high - df_low)
        df_SET = df_5WHL[["SET"]]
        df_5WHL = df_5WHL.iloc[-1].rename("52WHL")
        df_ROC = df.pct_change(10)*100
        df_ROC = df_ROC.iloc[-1].rename("ROC_10")

        df_close.index.rename("Name", inplace = True)
        df_high.index.rename("Name", inplace = True)
        df_low.index.rename("Name", inplace = True)
        df_5WHL.index.rename("Name", inplace = True)
        df_ROC.index.rename("Name", inplace = True)

        df_close.reset_index()
        df_high = df_high.reset_index()
        df_low = df_low.reset_index()
        df_5WHL = df_5WHL.reset_index()
        df_ROC = df_ROC.reset_index()

        mkt_cap_tab2 = mc.copy()
        mkt_cap_tab2["MarketCap"] = (mkt_cap_tab2["MarketCap"]/1000000)
        mkt_cap_tab2.rename(columns = {"MarketCap": "MarketCap (million THB)"}, inplace = True)
        sector_sub = raw_std[["symbol","market" ,"sector", "sub-sector"]]

        df_plot = pd.merge(left = df_high, right = df_low,
                           left_on= "Name", right_on = "Name",
                           how = "left").merge(right = df_close, left_on= "Name", right_on= "Name",
                           how = "left").merge(right = df_5WHL, left_on= "Name", right_on = "Name",
                           how = "left").merge(right = df_ROC, left_on= "Name", right_on = "Name",
                           how = "left").merge(right = mkt_cap_tab2, left_on= "Name", right_on = "Symbols",
                           how = "left").merge(right = sector_sub, left_on= "Name", right_on = "symbol",
                           how = "left")
        df_plot["MarketCap (million THB)"] = df_plot["MarketCap (million THB)"]

        df_SET = df_plot[df_plot["market"] == "SET"]
        df_mai = df_plot[df_plot["market"] == "mai"]


        return df_SET, df_mai


    _52SET, _52mai = _52WHL(dp)
    source_52SET = _52SET.drop(["Symbols", "MarketCap (million THB)", "symbol", "market", "sector", "sub-sector"], axis = 1)
    source_52SET.attrs['Description'] = "Technical Analysis with 52 Weeks High Low and Rate of Change in 10 Days also included the current High Low Close Price"
    source_52mai = _52mai.drop(["Symbols", "MarketCap (million THB)", "symbol", "market", "sector", "sub-sector"], axis = 1)
    source_52mai.attrs['Description'] = "Technical Analysis with 52 Weeks High Low and Rate of Change in 10 Days also included the current High Low Close Price"

    # st.write("source_52SET") ################################################ data source ################################################
    # st.write("source_52mai") ################################################ data source ################################################

    def plot_52WHL(data, select_sector, select_sub):
        data = data[data["sector"] == select_sector]
        if select_sub != None:
            data = data[data["sub-sector"] == select_sub]
        fig_52WHL = px.scatter(data, x = "ROC_10", y = "52WHL",
                               color = "Name", size = "MarketCap (million THB)",
                               title = f"52 Weeks High Low vs Rate of Change 10:<br> &nbsp;Sector: <i>{select_sector}<i> <br> &nbsp;Sub-Sector: <i>{select_sub}</i>",
                               text = "Name",
                               custom_data= ["Name", "ROC_10", "52WHL","sector" ,"sub-sector","MarketCap (million THB)", 'Close'],
                               template = "seaborn")

        fig_52WHL.add_vline(x=0, line_color='red', line_width=2, opacity=0.6)
        fig_52WHL.add_hline(y=0, line_color='red', line_width=2, opacity=0.6)
        fig_52WHL.add_hline(y=1, line_color='red', line_width=2, opacity=0.6)

        # update details in the hover
        fig_52WHL.update_traces(
        hovertemplate="<b>Name:</b> %{customdata[0]}<br>" +
                      "<b>Close:</b> %{customdata[6]:,.2f} <br>" +
                      "<b>ROC_10:</b> %{customdata[1]:,.2f}% <br>"
                      "<b>52WHL:</b> %{customdata[2]:,.2f} <br>" +
                      "<b>sector:</b> %{customdata[3]} <br>"
                      "<b>sub-sector:</b> %{customdata[4]} <br>" +
                      "<b>MarketCap:</b> %{customdata[5]:,.0f} million THB <br>",
        textposition = "bottom center",)

        fig_52WHL.update_layout(
            hovermode= "closest"
            )

        # fig_52WHL.update_xaxes(dtick=5)
        # fig_52WHL.update_yaxes(dtick=5)

        return fig_52WHL

    SET_t2,mai_t2 = geo_calculation(start)
    perf_source_SET = pd.pivot(SET_t2, index = "Name", columns= "Date", values= "Return").reset_index()
    perf_source_mai = pd.pivot(mai_t2, index = "Name", columns= "Date", values= "Return").reset_index()
    perf_source_SET.attrs["description"] = 'This dataset contains performance data within 100 days.'
    perf_source_mai.attrs["description"] = 'This dataset contains performance data within 100 days.'
    # st.write("perf_source_SET") ################################################ data source ################################################
    # st.write("perf_source_mai") ################################################ data source ################################################

    tcol1, tcol2 = st.columns(2)
    st.divider()
    tcol3, tcol4 = st.columns(2)
    with tcol1:
        st.subheader("SET Performance 100 Days")

        stcol1,stcol2 = st.columns(2)
        with stcol1:
            SetSectorSeclect_1_tab2 = st.radio("Select Sector", sorted(raw_std[raw_std["market"] == "SET"]['sector'].unique()), horizontal= True, key = "tab2 SET radio1")
        with stcol2:
            SetSectorSeclect_2_tab2 = st.radio("Select Sub-Sector",
                                           sorted(raw_std[raw_std["sector"] == SetSectorSeclect_1_tab2]["sub-sector"].dropna().unique()),
                                           horizontal= True, label_visibility= "collapsed", key = "tab2 SET radio2") if st.checkbox("include sub-sector", key = "tab2 SET checkbox1") else None\
                                           # st.radio("Select Sub-Sector",
                                           # sorted(raw_std[raw_std["sector"] == SetSectorSeclect_1]["sub-sector"].dropna().unique()),
                                           # horizontal= True, label_visibility= "collapsed", disabled= True, key = "tab2 SET radio2")

        st.plotly_chart(plot_geo(SET_t2, SetSectorSeclect_1_tab2, SetSectorSeclect_2_tab2))
    with tcol2:
        st.subheader("SET 52 Weeks High-Low vs Rate of Change 10 Days")
        st.plotly_chart(plot_52WHL(_52SET, SetSectorSeclect_1_tab2, SetSectorSeclect_2_tab2))

    with tcol3:
        st.subheader("mai Performance 100 Days")
        stcol3,stcol4 = st.columns(2)
        with stcol3:
            maiSectorSeclect_1_tab2 = st.radio("Select Sector", sorted(raw_std[raw_std["market"] == "mai"]['sector'].unique()), horizontal= True, key = "tab2 mai radio1")

        st.plotly_chart(plot_geo(mai_t2, maiSectorSeclect_1_tab2, None))
    with tcol4:
        st.subheader("mai 52 Weeks High-Low vs Rate of Change 10 Days")
        st.plotly_chart(plot_52WHL(_52mai, maiSectorSeclect_1_tab2, None))

    st.divider()

with tab3:
    from plotly.subplots import make_subplots

    def render_volume_charts(market_name):
        st.subheader(f"{market_name.upper()} Sector & Stock Volume Analysis")
        
        # Sector Analysis
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("###### Sector Volume")
            period_sector = st.radio("Choose Period (Days)", [30, 60, 90], key=f"period_sector_{market_name}", horizontal=True)
            
            temp_market = raw_std[raw_std["market"] == market_name]
            sectors_for_subplot = sorted(temp_market["sector"].unique())
            fig_VA = make_subplots(rows=len(sectors_for_subplot), cols=1, subplot_titles=sectors_for_subplot)

            for i, sector in enumerate(sectors_for_subplot):
                temp_sector_df = temp_market[temp_market["sector"] == sector]
                temp_tickers = sorted(temp_sector_df["symbol"].unique())
                
                temp_mc = mc[mc["Symbols"].isin(temp_tickers)].copy()
                if temp_mc["MarketCap"].sum() > 0:
                    temp_mc["weight"] = temp_mc["MarketCap"] / temp_mc["MarketCap"].sum()
                else:
                    temp_mc["weight"] = 0

                VA = dv[temp_tickers].tail(period_sector + 70).copy().fillna(0)
                weighted_va = VA.mul(temp_mc.set_index('Symbols')['weight'], axis=1)
                
                # --- [FIXED] Use groupby().sum() for 1-dimensional result ---
                sector_sum = weighted_va.sum(axis=1)
                
                threshold = sector_sum.rolling(window=22).mean() + 2 * sector_sum.rolling(window=22).std()
                anomaly = sector_sum > threshold
                
                plot_df = pd.DataFrame({'Date': sector_sum.index, 'Sum': sector_sum.values, 'Anomaly': anomaly.values}).tail(period_sector)

                fig_VA.add_trace(go.Bar(
                    x=plot_df["Date"], y=plot_df["Sum"],
                    name=sector, showlegend=False,
                    marker=dict(color=plot_df["Anomaly"].map({True: "orange", False: "gray"}))
                ), row=i + 1, col=1)
            
            fig_VA.update_layout(height=max(600, len(sectors_for_subplot) * 150), title=f"{market_name} Sector Volume Analysis")
            st.plotly_chart(fig_VA, use_container_width=True)

        # Stock Analysis
        with c2:
            st.markdown("###### Stock Volume (by Sector)")
            selected_sector_stock = st.selectbox("Select Sector to Analyze Stocks", options=sorted(temp_market["sector"].unique()), key=f"sector_stock_{market_name}")
            
            period_stock = st.radio("Choose Period (Days)", [30, 60, 90], key=f"period_stock_{market_name}", horizontal=True)
            
            temp_sector = temp_market[temp_market["sector"] == selected_sector_stock]
            symbols_for_subplot = sorted(temp_sector["symbol"].unique())
            
            if not symbols_for_subplot:
                st.warning("No stocks found in this sector.")
            else:
                fig_symbols = make_subplots(rows=len(symbols_for_subplot), cols=1, subplot_titles=symbols_for_subplot)
                
                vol_df = dv[symbols_for_subplot].tail(period_stock + 70).copy().fillna(0)
                vol_threshold = vol_df.rolling(window=22).mean() + 2 * vol_df.rolling(window=22).std()
                vol_anomaly = vol_df > vol_threshold
                
                for i, symbol in enumerate(symbols_for_subplot):
                    cutoff_for_plot = pd.DataFrame({
                        'Date': vol_df.index,
                        'Volume': vol_df[symbol],
                        'Anomaly': vol_anomaly[symbol]
                    }).tail(period_stock)
                    
                    fig_symbols.add_trace(go.Bar(
                        x=cutoff_for_plot["Date"], y=cutoff_for_plot["Volume"],
                        name=symbol, showlegend=False,
                        marker=dict(color=cutoff_for_plot["Anomaly"].map({True: "orange", False: "gray"}))
                    ), row=i + 1, col=1)

                fig_symbols.update_layout(height=max(600, len(symbols_for_subplot) * 150), title=f"Stock Volume for {selected_sector_stock} Sector")
                st.plotly_chart(fig_symbols, use_container_width=True)

    tab31, tab32 = st.tabs(["SET", "mai"])
    with tab31:
        render_volume_charts("SET")
    with tab32:
        render_volume_charts("mai")

with tab4:
    tab4c1, tab4c2 = st.columns([3,1])
    with tab4c2:
        hm_name = str.upper(st.text_input("Stock Name", "ADVANC"))
        st.success(f"กำลังวิเคราะห์หุ้น: {hm_name}")

    with tab4c1:
        if hm_name in dp.columns:
            hm_df = dp.copy()
            #fetch a stock
            hm_df = hm_df[[hm_name]].resample("M").last()
            hm_df["Return"] = hm_df.pct_change()
            hm_df = hm_df.dropna()
            hm_df = hm_df.reset_index()
            hm_df["Y"] = hm_df.Date.dt.strftime("%Y")
            hm_df["M"] = hm_df.Date.dt.strftime("%m")
            pivoted_hm_df = pd.pivot(hm_df, index = "Y", columns = "M", values = "Return")
            pivoted_hm_df.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug","Sep", "Oct", "Nov", "Dec"]
            pivoted_hm_df["Avg Y"] = pivoted_hm_df.mean(1)
            pivoted_hm_df.loc["Avg M"] = pivoted_hm_df.mean(0)

            plt.style.use("ggplot")
            fig, ax = plt.subplots(figsize=(10, 8)) # Increase figure size for better readability
            ax.set_facecolor('none')
            # Adjust font size with annot_kws
            heatmap = sns.heatmap(pivoted_hm_df.round(4) * 100, cmap="RdYlGn",annot=True, fmt=".2f", ax=ax, square= True, annot_kws= {"size":8}, center= 0, cbar=True, cbar_kws={'format': '%.0f%%'})

            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            ax.set_xlabel("Months", fontsize=10)
            ax.set_ylabel("Years", fontsize=10)
            plt.tight_layout()
            plt.title(f"Historical Monthly Return Heatmap of {hm_name} (%)", fontdict= {"size":12,"weight": "bold", "color": "#1A1A1A"})
            st.pyplot(fig)
        else:
            if hm_name != "":
                st.error(f"Stock '{hm_name}' not found in the database. Please try again.")
            else:
                pass

st.divider()

# Create a tuple of all the dataframes that will be passed to the AI functions
# This is done once to avoid repeating the list
# Dummy dataframes to avoid errors if VWAP is not calculated
SET_Stacked_bar, SET_VWAP, SET_VWAP_cumprod = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
mai_Stacked_bar, mai_VWAP, mai_VWAP_cumprod = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

data_for_ai_tuple = (
    rs_score_data, VP_change_D_data, VP_change_W_data, VP_change_M_data, VP_change_3M_data,
    source_52SET, source_52mai, perf_source_SET, perf_source_mai,
    SET_Stacked_bar, SET_VWAP, SET_VWAP_cumprod,
    mai_Stacked_bar, mai_VWAP, mai_VWAP_cumprod
)

# Prepare and store the main AI datasource in session_state once
if 'ai_datasource' not in st.session_state:
    from AI_sidebar import prepare_ai_datasource
    st.session_state.ai_datasource = prepare_ai_datasource(data_for_ai_tuple)

with st.sidebar:
    st.header("🤖 AI Technical Analyst")

    # Get the complete, unfiltered list of sectors and sub-sectors for the dropdowns
    all_markets = ["All"] + sorted(raw_std["market"].unique().tolist())
    all_sectors = ["All"] + sorted(raw_std["sector"].unique().tolist())
    all_sub_sectors = ["All"] + sorted(raw_std["sub-sector"].dropna().unique().tolist())

    # --- UI Filters ---
    selected_market = st.selectbox("Select Market", options=all_markets, key="market_select")

    # Dynamically filter sectors based on selected market
    if selected_market != "All":
        filtered_sectors = ["All"] + sorted(raw_std[raw_std["market"] == selected_market]["sector"].unique().tolist())
    else:
        filtered_sectors = all_sectors
    selected_sector = st.selectbox("Select Sector", options=filtered_sectors, key="sector_select")

    # Dynamically filter sub-sectors based on selected market and sector
    if selected_sector != "All":
        sub_sector_query = raw_std[raw_std["sector"] == selected_sector]
        if selected_market != "All":
            sub_sector_query = sub_sector_query[sub_sector_query["market"] == selected_market]
        filtered_sub_sectors = ["All"] + sorted(sub_sector_query["sub-sector"].dropna().unique().tolist())
    else:
        filtered_sub_sectors = all_sub_sectors
    selected_sub_sector = st.selectbox("Select Sub-Sector", options=filtered_sub_sectors, key="sub_sector_select")

    if st.button("🤖 วิเคราะห์ข้อมูลทางเทคนิค"):
        # Create a dynamic prompt based on the user's filter selections
        analysis_scope = "ภาพรวมตลาดทั้งหมด"
        if selected_market != "All":
            analysis_scope = f"ตลาด {selected_market}"
        if selected_sector != "All":
            analysis_scope += f" ในกลุ่มอุตสาหกรรม {selected_sector}"
        if selected_sub_sector != "All":
            analysis_scope += f" และหมวดย่อย {selected_sub_sector}"

        auto_prompt = f"""
        **Persona:** You are a seasoned Senior Technical Analyst providing insights to retail investors. Your tone should be clear, confident, and easy to understand. Avoid overly complex jargon. Your entire analysis must be based *only* on the provided CSV data.

        **Task:** Perform a technical analysis on the provided stock data for **{analysis_scope}**. Your goal is to identify the most noteworthy stock(s) and provide actionable insights.

        **Required Output Format (Strictly follow this structure and use Markdown):**

        ### 📊 บทวิเคราะห์ทางเทคนิคสำหรับ: {analysis_scope}

        **1. สรุปภาพรวม (Executive Summary):**
        * วิเคราะห์ภาพรวมของกลุ่มที่เลือกจากข้อมูล เช่น ค่าเฉลี่ยของ `Price %Change D`, `RS Score` เพื่อประเมิน Momentum ของกลุ่ม (แข็งแกร่ง, อ่อนแอ, หรือ Sideways).
        * ระบุแนวโน้มหลักที่เห็นจากข้อมูล.

        **2. ⭐ หุ้นที่โดดเด่นที่สุด (Top Pick):**
        * **ชื่อหุ้น:** [ระบุชื่อหุ้นที่น่าสนใจที่สุด 1 ตัวจากข้อมูลที่กรองมา]
        * **เหตุผลที่เลือก:** อธิบายโดยใช้ข้อมูลสนับสนุนอย่างน้อย 3 ข้อ.
            * **Momentum:** "มี Momentum แข็งแกร่งมาก สะท้อนจาก `RS Score` ที่ต่ำเพียง X และ `Rank` อยู่ที่ Y." (อธิบายว่าค่าต่ำคือดี).
            * **Price & Volume Action:** "ราคาหุ้นปรับตัวขึ้นอย่างมีนัยสำคัญ (`Price %Change D` อยู่ที่ Z%) พร้อมกับปริมาณการซื้อขายที่เพิ่มขึ้น (`Volume %Change D` อยู่ที่ A%) แสดงถึงความสนใจจากนักลงทุน."
            * **Trend Strength:** "อยู่ในแนวโน้มขาขึ้นที่ชัดเจน โดยราคายืนอยู่ใกล้จุดสูงสุดในรอบ 52 สัปดาห์ (`52WHL` = B) และมีแรงส่งระยะสั้น (`ROC_10` = C%)."
        * **คำแนะนำ:** "จากสัญญาณทางเทคนิคที่แข็งแกร่ง หุ้นตัวนี้น่าสนใจสำหรับการเก็งกำไรระยะสั้น-กลาง โดยมีแนวต้านถัดไปที่บริเวณ High เดิม และควรพิจารณาตัดขาดทุนหากราคาหลุดแนวรับสำคัญ."

        **3. 🏃 หุ้นที่น่าจับตามอง (Runner-Up / Other Noteworthy Stocks):**
        * **ชื่อหุ้น:** [ระบุชื่อหุ้นที่น่าสนใจรองลงมาอีก 1-2 ตัว ถ้ามี]
        * **สัญญาณที่น่าสนใจ:** "แสดงสัญญาณกลับตัวที่น่าสนใจ โดย `Price %Change D` เริ่มเป็นบวกและมี `Volume %Change D` เพิ่มขึ้นอย่างมีนัยสำคัญ" หรือ "แม้ `RS Score` จะยังไม่โดดเด่น แต่มี `Volume Trade Ratio` ที่สูง แสดงถึงสภาพคล่องและความสนใจของตลาด."

        **4. ⚠️ ข้อควรระวัง (Stocks to Watch for Weakness):**
        * **ชื่อหุ้น:** [ระบุชื่อหุ้นที่แสดงสัญญาณอ่อนแอที่สุด ถ้ามี]
        * **สัญญาณเตือน:** "แสดงสัญญาณอ่อนตัว โดยราคาปรับตัวลงแรง (`Price %Change D` ติดลบ) สวนทางกับตลาด และ `RS Score` อยู่ในระดับสูง แสดงถึง Momentum ที่อ่อนแอกว่าตลาดโดยรวม."

        **คำชี้แจง:** การวิเคราะห์นี้เป็นการวิเคราะห์ทางเทคนิคจากข้อมูลล่าสุดเท่านั้น ไม่ใช่การชี้นำการลงทุน โปรดใช้วิจารณญาณในการตัดสินใจลงทุน.
        """
        
        # Call the AI function with the selected filters
        response = get_ai_response(
            prompt=auto_prompt,
            market_filter=selected_market,
            sector_filter=selected_sector,
            sub_sector_filter=selected_sub_sector
        )
        st.markdown(response)

