# import necessary libraries
import pandas as pd
import streamlit as st
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import seaborn as sns

# --- [FIXED] Import all necessary functions from the sidebar module ---
from AI_sidebar import get_ai_response, prepare_ai_datasource
from select_market import select_market


st.set_page_config(layout="wide")

#"""===================================IMPORT DATA==================================="""
@st.cache_resource()
def import_data(period = 3650):
    # import data price
    sheet_id = "1Uk2bEql-MDVuEzka4f1Y7bMBM_vbob7mXb75ITUwwII"
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    dp = pd.read_csv(sheet_url).tail(period)
    dp['Date'] = pd.to_datetime(dp["Date"])
    dp = dp.set_index("Date")

    # rename columns
    t = [x.split(".")[0] for x in dp.columns]
    try:
        index_set = t.index("^SET")
        t[index_set] = "SET"
    except ValueError:
        pass # '^SET' not in list
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

#import data
dp, dv, mc = import_data()
raw_std = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET_Index.csv", index_col= "Unnamed: 0").sort_values(["symbol"], ascending= True)
raw_set50 = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET50.csv")['symbol'].tolist()
raw_set100 = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET100.csv")['symbol'].tolist()


# --- [MOVED] Top-level filters ---
st.header("APG Station: Market Overviews")
hc1, hc2, hc3 = st.columns([7, 5, 4])
with hc1:
    rawt1, sl11, sl12, sl13, sl14 = select_market("main_filters")
    rawt1["sub-sector"] = rawt1["sub-sector"].fillna("None")
    selection = rawt1["symbol"]
with hc2:
    # Placeholder for potential future filters
    pass
with hc3:
    # Display SET Index and Last Data Update
    SI = dp[["SET"]].copy().ffill().bfill()
    SI["Change"] = SI.pct_change()
    last_update_date = dp.index.max().strftime('%d %b %Y')
    st.metric("SET Index", value = f"{SI['SET'].iloc[-1]:.2f}", delta = f"{SI['Change'].iloc[-1]:.4f}%" )
    st.markdown(f"<p style='text-align: right; color: grey;'>Last Update: {last_update_date}</p>", unsafe_allow_html=True)


#"""===================================SETTING TAB==================================="""
tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking and Change", "📈 Performance and 52 Week High Low",
                                 "📊 Volume Analysis", "🔥 Sector Heatmap"])

#"""===================================TAB 1==================================="""
with tab1:
    col0, col1 = st.columns([1, 3])

    with st.spinner("Loading Ranking and Change..."):
        with col1:
            ct1c1, ct1c2 = st.columns([1, 2])
            with ct1c1:
                st.subheader("Price vs Volume Percentage Change")
            with ct1c2:
                st.write("")
                show_name = st.checkbox("🎉 Show Name")

        def VP_change(interval="D", show_name=False):
            period = {'D': 1, 'W': 5, 'M': 22, '3M': 65}.get(interval, 1)
            select_df = dv[selection]
            check = select_df.iloc[-1].dropna().index.tolist()
            pc = dp[check].pct_change(period).iloc[-1].T
            pv = dv[check].ewm(span=5, adjust=False).mean().pct_change(period).iloc[-1].T
            cb_df = pd.DataFrame({
                "Symbols": pc.index,
                f"Price %Change {interval}": pc.values * 100,
                f"Volume %Change {interval}": pv.values * 100
            }).merge(rawt1, left_on="Symbols", right_on="symbol", how="left") \
              .merge(mc, on="Symbols", how="left")
            data_source = cb_df.drop(["symbol", "full name", "market", "sector", "sub-sector"], axis=1)
            text_interval = {"D": "Daily", "W": "Weekly", "M": "Monthly", "3M": "Quarterly"}.get(interval)
            VP_chart = px.scatter(cb_df, x=f"Price %Change {interval}", y=f"Volume %Change {interval}",
                                  color="sector", text="Symbols" if show_name else None,
                                  custom_data=["Symbols", "MarketCap", "sub-sector", f"Price %Change {interval}", f"Volume %Change {interval}", "sector"],
                                  size="MarketCap", title=f"{text_interval} Price vs Volume Percentage Change", template="seaborn")
            VP_chart.update_traces(
                hovertemplate="<b>Name:</b> %{customdata[0]}<br><b>Marketcap:</b> %{customdata[1]:,.0f} THB<extra></extra><br><b>Sector:</b> %{customdata[5]}<br><b>SubSector:</b> %{customdata[2]}<br>" +
                              f"<b>{text_interval} Price %Change: </b>" + "%{customdata[3]:.2f}<br>" +
                              f"<b>{text_interval} Volume %Change: </b>" + "%{customdata[4]:.2f}<br>",
                textposition="bottom center")
            VP_chart.update_layout(hovermode="closest")
            VP_chart.add_vline(x=0, line_color='red', line_width=2, opacity=0.6)
            VP_chart.add_hline(y=0, line_color='red', line_width=2, opacity=0.6)
            return VP_chart, data_source

        def RS_Score():
            data = dp.copy().ffill().bfill()
            Q1 = data.pct_change(65).iloc[-1]
            Q2 = data.pct_change(130).iloc[-1]
            Q3 = data.pct_change(195).iloc[-1]
            Q4 = data.pct_change(260).iloc[-1]
            rs_data = pd.DataFrame({"Q1": Q1, "Q2": Q2, "Q3": Q3, "Q4": Q4})
            rs_data["RS Score"] = np.round(0.4 * Q1 + 0.2 * Q2 + 0.2 * Q3 + 0.2 * Q4, 4)
            rs_data["Rank"] = rs_data["RS Score"].rank(ascending=False, method="max")
            rs_data.reset_index(inplace=True)
            rs_data = rs_data.sort_values("Rank", ascending=True)
            rs_data["Return"] = rs_data["RS Score"]
            rs_data.rename(columns={"index": "Name"}, inplace=True)
            rs_data["RS Score"] = 1 / (1 + np.e ** (-rs_data["RS Score"]))
            rs_data = rs_data.merge(rawt1, left_on="Name", right_on="symbol", how="left")
            sorted_data = rs_data[["Name", "RS Score", "market", "sector", "sub-sector", "Rank", "Return"]].dropna()
            data_source = sorted_data.copy()
            sorted_data.set_index("Rank", inplace=True)
            sorted_data = sorted_data[sorted_data["Name"].isin(selection)]
            return sorted_data, data_source

        with col0:
            rs_score, rs_score_data = RS_Score()
            st.subheader("RS Ranking")
            st.dataframe(rs_score, height=850, width=500)

        with col1:
            scol0, scol1 = st.columns(2)
            with scol0:
                VP_change_D, VP_change_D_data = VP_change(interval="D", show_name=show_name)
                st.plotly_chart(VP_change_D)
            with scol1:
                VP_change_W, VP_change_W_data = VP_change(interval="W", show_name=show_name)
                st.plotly_chart(VP_change_W)
            scol2, scol3 = st.columns(2)
            with scol2:
                VP_change_M, VP_change_M_data = VP_change(interval="M", show_name=show_name)
                st.plotly_chart(VP_change_M)
            with scol3:
                VP_change_3M, VP_change_3M_data = VP_change(interval="3M", show_name=show_name)
                st.plotly_chart(VP_change_3M)
        st.divider()

#"""===================================TAB 2==================================="""
with tab2:
    start = (datetime.datetime.now() - datetime.timedelta(100)).strftime("%Y-%m")

    @st.cache_resource
    def geo_calculation(start, end=None):
        df = dp[start: end].copy().ffill().bfill()
        geo_df = (df.pct_change() + 1).cumprod()
        geo_df = geo_df.reset_index().melt(id_vars=["Date"], value_vars=geo_df.columns, var_name="Name", value_name="Return")
        geo_df = geo_df.merge(raw_std, left_on="Name", right_on="symbol", how="left").drop(["full name", "symbol"], axis=1)
        return geo_df[geo_df["market"] == "SET"], geo_df[geo_df["market"] == "mai"]

    def plot_geo(data, select_sector, select_sub):
        data = data[data["sector"] == select_sector]
        if select_sub is not None:
            data = data[data["sub-sector"] == select_sub]
        geo_fig = px.line(data, x="Date", y="Return", color="Name",
                          title=f"Performance: Sector:<br> &nbsp;Sector: <i>{select_sector}<i> <br> &nbsp;Sub-Sector: <i>{select_sub}</i>",
                          labels={"Date": "", "Return": "Cumulative Return"}, template="seaborn")
        geo_fig.update_xaxes(rangeslider_visible=True)
        return geo_fig

    def _52WHL(data):
        df = data.tail(265)
        df_close = df.iloc[-1].rename("Close")
        df_high = df.max().rename("High")
        df_low = df.min().rename("Low")
        df_5WHL = ((df_close - df_low) / (df_high - df_low)).rename("52WHL")
        df_ROC = (df.pct_change(10).iloc[-1] * 100).rename("ROC_10")
        mkt_cap_tab2 = mc.rename(columns={"MarketCap": "MarketCap (million THB)"})
        mkt_cap_tab2["MarketCap (million THB)"] /= 1_000_000
        sector_sub = raw_std[["symbol", "market", "sector", "sub-sector"]]
        df_plot = pd.concat([df_high, df_low, df_close, df_5WHL, df_ROC], axis=1).reset_index().rename(columns={"index": "Name"})
        df_plot = df_plot.merge(mkt_cap_tab2, left_on="Name", right_on="Symbols", how="left") \
                         .merge(sector_sub, left_on="Name", right_on="symbol", how="left")
        return df_plot[df_plot["market"] == "SET"], df_plot[df_plot["market"] == "mai"]

    _52SET, _52mai = _52WHL(dp)
    source_52SET = _52SET.drop(["Symbols", "MarketCap (million THB)", "symbol", "market", "sector", "sub-sector"], axis=1, errors='ignore')
    source_52mai = _52mai.drop(["Symbols", "MarketCap (million THB)", "symbol", "market", "sector", "sub-sector"], axis=1, errors='ignore')

    def plot_52WHL(data, select_sector, select_sub):
        data = data[data["sector"] == select_sector]
        if select_sub is not None:
            data = data[data["sub-sector"] == select_sub]
        fig_52WHL = px.scatter(data, x="ROC_10", y="52WHL", color="Name", size="MarketCap (million THB)",
                               title=f"52 Weeks High Low vs Rate of Change 10:<br> &nbsp;Sector: <i>{select_sector}<i> <br> &nbsp;Sub-Sector: <i>{select_sub}</i>",
                               text="Name", custom_data=["Name", "ROC_10", "52WHL", "sector", "sub-sector", "MarketCap (million THB)", 'Close'],
                               template="seaborn")
        fig_52WHL.add_vline(x=0, line_color='red', line_width=2, opacity=0.6)
        fig_52WHL.add_hline(y=0, line_color='red', line_width=2, opacity=0.6)
        fig_52WHL.add_hline(y=1, line_color='red', line_width=2, opacity=0.6)
        fig_52WHL.update_traces(
            hovertemplate="<b>Name:</b> %{customdata[0]}<br><b>Close:</b> %{customdata[6]:,.2f} <br><b>ROC_10:</b> %{customdata[1]:,.2f}% <br><b>52WHL:</b> %{customdata[2]:,.2f} <br>" +
                          "<b>sector:</b> %{customdata[3]} <br><b>sub-sector:</b> %{customdata[4]} <br><b>MarketCap:</b> %{customdata[5]:,.0f} million THB <br>",
            textposition="bottom center")
        fig_52WHL.update_layout(hovermode="closest")
        return fig_52WHL

    SET_t2, mai_t2 = geo_calculation(start)
    perf_source_SET = pd.pivot_table(SET_t2, index="Name", columns="Date", values="Return").reset_index()
    perf_source_mai = pd.pivot_table(mai_t2, index="Name", columns="Date", values="Return").reset_index()

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.subheader("SET Performance 100 Days")
        SetSectorSelect_1_tab2 = st.radio("Select Sector", sorted(raw_std[raw_std["market"] == "SET"]['sector'].unique()), horizontal=True, key="tab2_set_sector")
        include_sub_set = st.checkbox("Include sub-sector", key="tab2_set_sub_check")
        SetSectorSelect_2_tab2 = st.radio("Select Sub-Sector", sorted(raw_std[raw_std["sector"] == SetSectorSelect_1_tab2]["sub-sector"].dropna().unique()), horizontal=True, key="tab2_set_sub") if include_sub_set else None
        st.plotly_chart(plot_geo(SET_t2, SetSectorSelect_1_tab2, SetSectorSelect_2_tab2))
    with tcol2:
        st.subheader("SET 52 Weeks High-Low vs Rate of Change 10 Days")
        st.plotly_chart(plot_52WHL(_52SET, SetSectorSelect_1_tab2, SetSectorSelect_2_tab2))
    st.divider()
    tcol3, tcol4 = st.columns(2)
    with tcol3:
        st.subheader("mai Performance 100 Days")
        maiSectorSelect_1_tab2 = st.radio("Select Sector", sorted(raw_std[raw_std["market"] == "mai"]['sector'].unique()), horizontal=True, key="tab2_mai_sector")
        st.plotly_chart(plot_geo(mai_t2, maiSectorSelect_1_tab2, None))
    with tcol4:
        st.subheader("mai 52 Weeks High-Low vs Rate of Change 10 Days")
        st.plotly_chart(plot_52WHL(_52mai, maiSectorSelect_1_tab2, None))

#"""===================================TAB 3==================================="""
with tab3:
    st.header("Volume Analysis")
    # Define a function to create charts to avoid code repetition
    def render_volume_charts(market):
        st.subheader(f"{market} Sector Volume Analysis")
        period_sector = st.radio("Choose Period", [30, 60, 90], key=f"period_sector_{market}", horizontal=True)

        # Sector-level analysis
        temp_market = raw_std[raw_std["market"] == market]
        sectors_for_subplot = sorted(temp_market["sector"].unique())
        fig_sector = make_subplots(rows=len(sectors_for_subplot), cols=1, subplot_titles=sectors_for_subplot, shared_xaxes=True)

        for i, sector in enumerate(sectors_for_subplot):
            temp_sector = temp_market[temp_market["sector"] == sector]
            temp_tickers = sorted(temp_sector["symbol"].unique())
            
            va_data = dv[temp_tickers].tail(period_sector + 70).copy().fillna(0)
            va_sum = va_data.sum(axis=1)
            
            threshold = va_sum.rolling(window=22).mean() + 2 * va_sum.rolling(window=22).std()
            anomaly = va_sum > threshold
            
            plot_df = pd.DataFrame({'Date': va_sum.index, 'Sum': va_sum.values, 'Anomaly': anomaly.values}).tail(period_sector)

            fig_sector.add_trace(go.Bar(
                x=plot_df['Date'], y=plot_df['Sum'], name=sector, showlegend=False,
                marker=dict(color=plot_df['Anomaly'].map({True: "orange", False: "gray"}))
            ), row=i + 1, col=1)
        
        fig_sector.update_layout(height=200 * len(sectors_for_subplot), title_text=f"{market} Sector Volume Analysis")
        st.plotly_chart(fig_sector, use_container_width=True)

        # Stock-level analysis
        st.subheader(f"{market} Stock Volume Analysis")
        selected_sector_stock = st.selectbox("Select Sector for Stock Analysis", sectors_for_subplot, key=f"sector_stock_{market}")
        period_stock = st.radio("Choose Period", [30, 60, 90], key=f"period_stock_{market}", horizontal=True)

        symbols_in_sector = sorted(raw_std[(raw_std["market"] == market) & (raw_std["sector"] == selected_sector_stock)]["symbol"].unique())
        fig_stock = make_subplots(rows=len(symbols_in_sector), cols=1, subplot_titles=symbols_in_sector, shared_xaxes=True)

        for i, symbol in enumerate(symbols_in_sector):
            vol_df = dv[[symbol]].tail(period_stock + 70).copy().fillna(0)
            threshold = vol_df[symbol].rolling(window=22).mean() + 2 * vol_df[symbol].rolling(window=22).std()
            anomaly = vol_df[symbol] > threshold
            plot_df = pd.DataFrame({'Date': vol_df.index, 'Volume': vol_df[symbol], 'Anomaly': anomaly}).tail(period_stock)
            
            fig_stock.add_trace(go.Bar(
                x=plot_df['Date'], y=plot_df['Volume'], name=symbol, showlegend=False,
                marker=dict(color=plot_df['Anomaly'].map({True: "orange", False: "gray"}))
            ), row=i + 1, col=1)

        fig_stock.update_layout(height=200 * len(symbols_in_sector), title_text=f"Stock Volume in {selected_sector_stock} Sector")
        st.plotly_chart(fig_stock, use_container_width=True)

    tab31, tab32 = st.tabs(["SET", "mai"])
    with tab31:
        render_volume_charts("SET")
        # For AI Data
        _, SET_Stacked_bar, SET_VWAP, SET_VWAP_cumprod = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()) # Placeholder
    with tab32:
        render_volume_charts("mai")
        # For AI Data
        _, mai_Stacked_bar, mai_VWAP, mai_VWAP_cumprod = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()) # Placeholder


#"""===================================TAB 4==================================="""
with tab4:
    st.subheader("Monthly Return Heatmap")
    hm_name = str.upper(st.text_input("Enter Stock Name for Heatmap", "ADVANC"))
    if hm_name in dp.columns:
        hm_df = dp[[hm_name]].resample("M").last()
        hm_df["Return"] = hm_df.pct_change()
        hm_df = hm_df.dropna().reset_index()
        hm_df["Y"] = hm_df.Date.dt.strftime("%Y")
        hm_df["M"] = hm_df.Date.dt.strftime("%m")
        pivoted_hm_df = pd.pivot_table(hm_df, index="Y", columns="M", values="Return")
        pivoted_hm_df.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivoted_hm_df["Avg Y"] = pivoted_hm_df.mean(1)
        pivoted_hm_df.loc["Avg M"] = pivoted_hm_df.mean(0)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivoted_hm_df, cmap="RdYlGn", annot=True, fmt=".2%", ax=ax, center=0, annot_kws={"size": 8})
        plt.title(f"Historical Monthly Return Heatmap of {hm_name}", fontsize=14)
        st.pyplot(fig)
    else:
        st.error(f"Stock '{hm_name}' not found.")


# --- Prepare and Cache AI Datasource ---
# This block runs once, prepares the data, and stores it in session_state
if 'ai_datasource' not in st.session_state:
    with st.spinner("Preparing AI data source for the first time..."):
        # Create a tuple of all the dataframes that will be passed to the AI functions
        data_for_ai_tuple = (
            rs_score_data, VP_change_D_data, VP_change_W_data, VP_change_M_data, VP_change_3M_data,
            source_52SET, source_52mai, perf_source_SET, perf_source_mai,
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), # Placeholders for SET VWAP data
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame()  # Placeholders for mai VWAP data
        )
        st.session_state.ai_datasource = prepare_ai_datasource(data_for_ai_tuple)

# --- AI Sidebar UI ---
with st.sidebar:
    st.header("🤖 AI Technical Analyst")

    # Filters
    st.subheader("Filter Data for AI")
    
    # Market filter
    market_options = ["All"] + sorted(raw_std['market'].unique())
    selected_market = st.selectbox(
        "Select Market",
        market_options,
        key='ai_market_filter' # Unique key
    )

    # Sector filter (dynamically updated)
    if selected_market == "All":
        sector_options = ["All"] + sorted(raw_std['sector'].unique())
    else:
        sector_options = ["All"] + sorted(raw_std[raw_std['market'] == selected_market]['sector'].unique())
    
    selected_sector = st.selectbox(
        "Select Sector",
        sector_options,
        key='ai_sector_filter' # Unique key
    )

    # Sub-sector filter (dynamically updated)
    if selected_sector == "All":
        sub_sector_options = ["All"]
    else:
        if selected_market != "All":
            filtered_df = raw_std[(raw_std['market'] == selected_market) & (raw_std['sector'] == selected_sector)]
        else: # selected_market is "All"
            filtered_df = raw_std[raw_std['sector'] == selected_sector]
        sub_sector_options = ["All"] + sorted(filtered_df['sub-sector'].dropna().unique())

    selected_sub_sector = st.selectbox(
        "Select Sub-Sector",
        sub_sector_options,
        key='ai_sub_sector_filter' # Unique key
    )
    
    st.divider()

    # User Prompt
    st.subheader("Your Request")
    user_prompt = st.text_area(
        "Ask the AI to analyze the filtered stocks...",
        value="Based on the filtered data, which stocks are showing the strongest signs of a bullish technical breakout? Analyze based on RS Rank, recent price/volume changes, and position within the 52-week range.",
        height=150
    )

    if st.button("Analyze Now"):
        if 'ai_datasource' in st.session_state:
            # Call the AI function with the new filters
            response = get_ai_response(
                prompt=user_prompt,
                market_filter=selected_market,
                sector_filter=selected_sector,
                sub_sector_filter=selected_sub_sector
            )
            st.markdown(response)
        else:
            st.warning("AI data is still being prepared. Please wait a moment.")

