# import necessary libraries
import pandas as pd
import streamlit as st
import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
from select_market import select_market
from AI_sidebar import get_ai_response # Import the updated function

st.set_page_config(layout="wide")

#"""===================================IMPORT DATA==================================="""
@st.cache_resource()
def import_data(period = 3650):
    # import data price
    import pandas as pd

    # 1. สร้าง URL สำหรับดาวน์โหลดเป็นไฟล์ CSV
    sheet_id = "1Uk2bEql-MDVuEzka4f1Y7bMBM_vbob7mXb75ITUwwII"
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

    # 2. อ่านข้อมูลจาก URL ด้วย pandas และเลือกเฉพาะท้ายตารางตามตัวแปร period
    dp = pd.read_csv(sheet_url).tail(period)

    # 3. แปลงคอลัมน์ Date ให้เป็น Datetime และตั้งเป็น Index
    dp['Date'] = pd.to_datetime(dp["Date"])
    dp = dp.set_index("Date")

    # rename columns
    t = [x.split(".")[0] for x in dp.columns]
    try:
        index_set = t.index("^SET")
        t[index_set] = "SET"
    except ValueError:
        pass # Handle case where '^SET' is not in the list
    dp.columns = t

    # import data volume
    dv = pd.read_csv("source_volume.csv",).tail(3650)
    dv.Date = pd.to_datetime(dv['Date'])
    dv = dv.set_index("Date")
    dv.columns = t

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

# Main app layout
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

# --- All data sources are prepared here once ---
with st.spinner("Preparing data for analysis..."):
    # TAB 1 Data
    rawt1,_,_,_,_ = select_market("tab1")
    rawt1["sub-sector"] = rawt1["sub-sector"].fillna("None")
    selection = rawt1["symbol"]
    
    # RS Score
    data_rs = dp.copy().ffill().bfill()
    Q1 = data_rs.pct_change(65).iloc[-1]
    Q2 = data_rs.pct_change(130).iloc[-1]
    Q3 = data_rs.pct_change(195).iloc[-1]
    Q4 = data_rs.pct_change(260).iloc[-1]
    rs_data = pd.DataFrame({"Q1": Q1, "Q2": Q2, "Q3": Q3, "Q4": Q4})
    rs_data["RS Score"] = np.round(0.4*Q1 + 0.2*Q2 + 0.2*Q3 + 0.2*Q4,4)
    rs_data["Rank"] = rs_data["RS Score"].rank(ascending=False, method="max")
    rs_data.reset_index(inplace=True)
    rs_data = rs_data.rename(columns={"index": "Name"})
    rs_data["Return"] = rs_data["RS Score"]
    rs_data["RS Score"] = 1 / (1 + np.e**(-rs_data["RS Score"]))
    rs_score_data = rs_data.merge(rawt1, left_on="Name", right_on="symbol", how="left")[["Name", "RS Score", "market", "sector", "sub-sector", "Rank", "Return"]]
    rs_score_data.dropna(inplace=True)

    # VP Change Data
    def get_vp_change_data(interval):
        period_map = {"D": 1, "W": 5, "M": 22, "3M": 65}
        period = period_map.get(interval, 1)
        # Ensure 'selection' is not empty and symbols exist in 'dv'
        valid_selection = [s for s in selection if s in dv.columns]
        if not valid_selection:
            return pd.DataFrame() # Return empty if no valid stocks
        check = dv[valid_selection].iloc[-1].dropna().index.tolist()
        if not check:
            return pd.DataFrame() # Return empty if no data after dropping NA
        pc = dp[check].pct_change(period).iloc[-1]
        pv = dv[check].ewm(span=5, adjust=False).mean().pct_change(period).iloc[-1]
        cb_df = pd.DataFrame({"Symbols": pc.index, f"Price %Change {interval}": pc.values*100, f"Volume %Change {interval}": pv.values*100})
        cb_df = cb_df.merge(mc, on="Symbols", how="left")
        return cb_df
    
    VP_change_D_data = get_vp_change_data("D")
    VP_change_W_data = get_vp_change_data("W")
    VP_change_M_data = get_vp_change_data("M")
    VP_change_3M_data = get_vp_change_data("3M")

    # TAB 2 Data
    start_date = (datetime.datetime.now() - datetime.timedelta(100)).strftime("%Y-%m")
    df_perf = dp[start_date:].copy().ffill().bfill()
    geo_df = (df_perf.pct_change()+1).cumprod().reset_index().melt(id_vars=["Date"], var_name="Name", value_name="Return")
    geo_df = geo_df.merge(raw_std, left_on="Name", right_on="symbol", how="left").drop(["full name", "symbol"], axis=1)
    SET_t2 = geo_df[geo_df["market"] == "SET"]
    mai_t2 = geo_df[geo_df["market"] == "mai"]
    perf_source_SET = pd.pivot_table(SET_t2, index="Name", columns="Date", values="Return").reset_index()
    perf_source_mai = pd.pivot_table(mai_t2, index="Name", columns="Date", values="Return").reset_index()

    df_52 = dp.tail(265)
    df_close = df_52.iloc[-1].rename("Close")
    df_high = df_52.max().rename("High")
    df_low = df_52.min().rename("Low")
    df_5WHL = ((df_52 - df_low) / (df_high - df_low)).iloc[-1].rename("52WHL")
    df_ROC = (df_52.pct_change(10) * 100).iloc[-1].rename("ROC_10")
    mkt_cap_tab2 = mc.copy()
    mkt_cap_tab2["MarketCap (million THB)"] = (mkt_cap_tab2["MarketCap"] / 1000000)
    
    df_plot = pd.concat([df_high, df_low, df_close, df_5WHL, df_ROC], axis=1).reset_index().rename(columns={'index':'Name'})
    df_plot = df_plot.merge(mkt_cap_tab2, left_on="Name", right_on="Symbols", how="left")
    df_plot = df_plot.merge(raw_std[["symbol", "market", "sector", "sub-sector"]], left_on="Name", right_on="symbol", how="left")
    _52SET = df_plot[df_plot["market"] == "SET"]
    _52mai = df_plot[df_plot["market"] == "mai"]
    source_52SET = _52SET.drop(["Symbols", "MarketCap (million THB)", "symbol", "market", "sector", "sub-sector", "MarketCap"], axis=1, errors='ignore')
    source_52mai = _52mai.drop(["Symbols", "MarketCap (million THB)", "symbol", "market", "sector", "sub-sector", "MarketCap"], axis=1, errors='ignore')

    # TAB 3 Data
    def get_vwap_data(market, sector=None, period=90):
        temp_market = raw_std[raw_std["market"] == market]
        symbols = sorted(temp_market["symbol"].unique()) if sector is None else sorted(temp_market[temp_market["sector"] == sector]["symbol"].unique())
        # Filter symbols that exist in dp and dv to prevent KeyErrors
        valid_symbols = [s for s in symbols if s in dp.columns and s in dv.columns]
        if not valid_symbols:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        price = dp[valid_symbols].tail(period*2).copy().ffill().bfill()
        vol = dv[valid_symbols].tail(period*2).copy().ffill().bfill()
        vwap = (price * vol).cumsum() / vol.cumsum()
        vwap_data = vwap.T.reset_index().rename(columns={"index": "Name"})
        vwap_cumprod = (vwap.pct_change()+1).cumprod()
        vwap_cumprod_data = vwap_cumprod.T.reset_index().rename(columns={"index": "Name"})
        
        df_stack = dv[valid_symbols].tail(period*2).copy().ffill().bfill()
        df_stacked_bar = df_stack * vwap
        df_stacked_bar["sum"] = df_stacked_bar.sum(1)
        df_stacked_bar_ratio = df_stacked_bar.iloc[:, :-1].div(df_stacked_bar["sum"], axis = 0)
        stacked_bar_ratio_datasource = df_stacked_bar_ratio.T
        return stacked_bar_ratio_datasource, vwap_data, vwap_cumprod_data
        
    SET_Stacked_bar, SET_VWAP, SET_VWAP_cumprod = get_vwap_data("SET")
    mai_Stacked_bar, mai_VWAP, mai_VWAP_cumprod = get_vwap_data("mai")
    
#"""===================================TAB 1 RENDER==================================="""
with tab1:
    col0, col1 = st.columns([1,3])
    with col0:
        st.subheader("RS Ranking")
        df_to_show = rs_score_data.copy()
        df_to_show = df_to_show[df_to_show["Name"].isin(selection)]
        st.dataframe(df_to_show.set_index("Rank"), height=850, width=500)
    
    with col1:
        ct1c1, ct1c2 = st.columns([1,2])
        with ct1c1:
            st.subheader("Price vs Volume Percentage Change")
        with ct1c2:
            st.write("")
            show_name = st.checkbox("🎉 Show Name")
        
        def plot_vp_change(interval, data):
            if data.empty:
                return go.Figure().update_layout(title_text=f"No data available for {interval} VP Change")
            text_interval = "Daily" if interval == "D" else "Weekly" if interval == "W" else "Monthly" if interval == "M" else "Quarterly"
            df_use = data.merge(rawt1, left_on="Symbols", right_on="symbol", how="left")
            df_use = df_use[df_use["Symbols"].isin(selection)]
            
            fig = px.scatter(df_use, x=f"Price %Change {interval}", y=f"Volume %Change {interval}",
                               color="sector", text="Symbols" if show_name else None,
                               size="MarketCap", title=f"{text_interval} Price vs Volume Percentage Change",
                               custom_data=["Symbols", "MarketCap", "sub-sector", f"Price %Change {interval}", f"Volume %Change {interval}", "sector"],
                               template="seaborn")
            fig.update_traces(
                hovertemplate="<b>Name:</b> %{customdata[0]}<br>" +
                              "<b>Marketcap:</b> %{customdata[1]:,.0f} THB<extra></extra><br>" +
                              "<b>Sector:</b> %{customdata[5]}<br>"+
                              "<b>SubSector:</b> %{customdata[2]}<br>"+
                              f"<b>{text_interval} Price %Change: </b>" "%{customdata[3]:.2f}<br>"+
                              f"<b>{text_interval} Volume %Change: </b>" "%{customdata[4]:.2f}<br>",
                textposition="bottom center"
            )
            fig.add_vline(x=0, line_color='red', line_width=2, opacity=0.6)
            fig.add_hline(y=0, line_color='red', line_width=2, opacity=0.6)
            return fig

        scol0, scol1 = st.columns(2)
        with scol0:
            st.plotly_chart(plot_vp_change("D", VP_change_D_data))
        with scol1:
            st.plotly_chart(plot_vp_change("W", VP_change_W_data))
        
        scol2, scol3 = st.columns(2)
        with scol2:
            st.plotly_chart(plot_vp_change("M", VP_change_M_data))
        with scol3:
            st.plotly_chart(plot_vp_change("3M", VP_change_3M_data))

#"""===================================TAB 2 RENDER==================================="""
with tab2:
    def plot_geo(data, select_sector, select_sub):
        data = data[data["sector"] == select_sector]
        if select_sub:
            data = data[data["sub-sector"] == select_sub]
        fig = px.line(data, x="Date", y="Return", color="Name",
                        title=f"Performance:<br> &nbsp;Sector: <i>{select_sector}<i> <br> &nbsp;Sub-Sector: <i>{select_sub or 'All'}</i>",
                        labels={"Date": "", "Return": "Cumulative Return"}, template="seaborn")
        return fig
    
    def plot_52WHL(data, select_sector, select_sub):
        data = data[data["sector"] == select_sector]
        if select_sub:
            data = data[data["sub-sector"] == select_sub]
        fig = px.scatter(data, x="ROC_10", y="52WHL", color="Name", size="MarketCap (million THB)",
                           title=f"52 Weeks High Low vs Rate of Change 10:<br> &nbsp;Sector: <i>{select_sector}<i> <br> &nbsp;Sub-Sector: <i>{select_sub or 'All'}</i>",
                           text="Name",
                           custom_data=["Name", "ROC_10", "52WHL", "sector", "sub-sector", "MarketCap (million THB)", 'Close'],
                           template="seaborn")
        fig.add_vline(x=0, line_color='red', line_width=2, opacity=0.6)
        fig.add_hline(y=0, line_color='red', line_width=2, opacity=0.6)
        fig.add_hline(y=1, line_color='red', line_width=2, opacity=0.6)
        fig.update_traces(
            hovertemplate="<b>Name:</b> %{customdata[0]}<br>" +
                          "<b>Close:</b> %{customdata[6]:,.2f} <br>" +
                          "<b>ROC_10:</b> %{customdata[1]:,.2f}% <br>"
                          "<b>52WHL:</b> %{customdata[2]:,.2f} <br>" +
                          "<b>sector:</b> %{customdata[3]} <br>"
                          "<b>sub-sector:</b> %{customdata[4]} <br>" +
                          "<b>MarketCap:</b> %{customdata[5]:,.0f} million THB <br>",
            textposition="bottom center")
        return fig

    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.subheader("SET Performance & 52 Week Analysis")
        set_sectors = sorted(raw_std[raw_std["market"] == "SET"]['sector'].unique())
        set_sector_select = st.selectbox("Select SET Sector", set_sectors, key="set_sector_t2")
        set_sub_sectors = sorted(raw_std[(raw_std["sector"] == set_sector_select) & (raw_std["market"] == "SET")]["sub-sector"].dropna().unique())
        set_sub_select = st.selectbox("Select SET Sub-Sector (Optional)", ["All"] + set_sub_sectors, key="set_sub_t2")
        set_sub_select = None if set_sub_select == "All" else set_sub_select
        st.plotly_chart(plot_geo(SET_t2, set_sector_select, set_sub_select))
        st.plotly_chart(plot_52WHL(_52SET, set_sector_select, set_sub_select))
        
    with tcol2:
        st.subheader("mai Performance & 52 Week Analysis")
        mai_sectors = sorted(raw_std[raw_std["market"] == "mai"]['sector'].unique())
        mai_sector_select = st.selectbox("Select mai Sector", mai_sectors, key="mai_sector_t2")
        st.plotly_chart(plot_geo(mai_t2, mai_sector_select, None))
        st.plotly_chart(plot_52WHL(_52mai, mai_sector_select, None))

#"""===================================TAB 3 RENDER==================================="""
with tab3:
    from plotly.subplots import make_subplots
    st.subheader("Volume Analysis")
    
    # Render Function for Volume Charts
    def render_volume_charts(market):
        sectors = sorted(raw_std[raw_std["market"] == market]['sector'].unique())
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{market} Sector Volume**")
            period_sector = st.radio("Choose Period", [30, 60, 90], key=f"period_sector_{market}", horizontal=True)
            
            fig_sector = make_subplots(rows=len(sectors), cols=1, subplot_titles=sectors)
            for i, sector in enumerate(sectors):
                tickers = sorted(raw_std[(raw_std["sector"] == sector) & (raw_std["market"] == market)]["symbol"].unique())
                if not tickers: continue
                
                temp_mc = mc[mc["Symbols"].isin(tickers)]
                temp_mc["weight"] = temp_mc["MarketCap"] / temp_mc["MarketCap"].sum()
                
                va_df = dv[tickers].tail(period_sector + 70).copy().fillna(0).reset_index()
                va_df = pd.melt(va_df, id_vars=["Date"], var_name="Name", value_name="Volume")
                va_df = va_df.merge(temp_mc, left_on="Name", right_on="Symbols").drop("Symbols", axis=1)
                va_df["Weighted_Volume"] = va_df["Volume"] * va_df["weight"]
                
                # --- [FIXED] Use groupby().sum() to ensure a 1D Series is returned ---
                pivot_va = va_df.groupby('Date')['Weighted_Volume'].sum()
                threshold = pivot_va.rolling(22).mean() + 2 * pivot_va.rolling(22).std()
                anomaly = pivot_va > threshold
                
                plot_df = pd.DataFrame({'Date': pivot_va.index, 'Sum': pivot_va.values, 'Anomaly': anomaly.values}).tail(period_sector)
                
                fig_sector.add_trace(go.Bar(x=plot_df["Date"], y=plot_df["Sum"], name=sector, showlegend=False,
                                            marker=dict(color=plot_df["Anomaly"].map({True: "orange", False:"gray"}))),
                                     row=i+1, col=1)
            fig_sector.update_layout(height=300*len(sectors), title_text=f"{market} Sector Volume Analysis")
            st.plotly_chart(fig_sector, use_container_width=True)

        with c2:
            st.markdown(f"**{market} Stock Volume**")
            selected_sector_stock = st.selectbox("Select Sector", sectors, key=f"sector_stock_{market}")
            period_stock = st.radio("Choose Period", [30, 60, 90], key=f"period_stock_{market}", horizontal=True)
            
            symbols_in_sector = sorted(raw_std[(raw_std["sector"] == selected_sector_stock) & (raw_std["market"] == market)]["symbol"].unique())
            if symbols_in_sector:
                fig_stock = make_subplots(rows=len(symbols_in_sector), cols=1, subplot_titles=symbols_in_sector)
                for i, symbol in enumerate(symbols_in_sector):
                    vol_df = dv[[symbol]].tail(period_stock + 70).copy().fillna(0)
                    threshold = vol_df[symbol].rolling(22).mean() + 2 * vol_df[symbol].rolling(22).std()
                    anomaly = vol_df[symbol] > threshold
                    plot_df = pd.DataFrame({'Date': vol_df.index, 'Volume': vol_df[symbol].values, 'Anomaly': anomaly.values}).tail(period_stock)
                    
                    fig_stock.add_trace(go.Bar(x=plot_df["Date"], y=plot_df["Volume"], name=symbol, showlegend=False,
                                                marker=dict(color=plot_df["Anomaly"].map({True: "orange", False:"gray"}))),
                                        row=i+1, col=1)
                fig_stock.update_layout(height=200*len(symbols_in_sector), title_text=f"Stock Volume for {selected_sector_stock} Sector")
                st.plotly_chart(fig_stock, use_container_width=True)
            else:
                st.info(f"No stocks found for sector '{selected_sector_stock}' in {market} market.")

    set_tab, mai_tab = st.tabs(["SET", "mai"])
    with set_tab:
        render_volume_charts("SET")
    with mai_tab:
        render_volume_charts("mai")

#"""===================================TAB 4 RENDER==================================="""
with tab4:
    st.subheader("Monthly Return Heatmap")
    tab4c1, tab4c2 = st.columns([3,1])
    with tab4c2:
        hm_name = st.text_input("Stock Name", "ADVANC").upper()

    with tab4c1:
        if hm_name in dp.columns:
            hm_df = dp[[hm_name]].copy().resample("M").last()
            hm_df["Return"] = hm_df.pct_change()
            hm_df.dropna(inplace=True)
            hm_df["Y"] = hm_df.index.strftime("%Y")
            hm_df["M"] = hm_df.index.strftime("%m")
            
            pivoted_hm_df = pd.pivot_table(hm_df, index="Y", columns="M", values="Return")
            pivoted_hm_df.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug","Sep", "Oct", "Nov", "Dec"]
            pivoted_hm_df["Avg Y"] = pivoted_hm_df.mean(1)
            pivoted_hm_df.loc["Avg M"] = pivoted_hm_df.mean(0)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(pivoted_hm_df, cmap="RdYlGn", annot=True, fmt=".2%", ax=ax,
                        annot_kws={"size": 8}, center=0)
            ax.set_title(f"Historical Monthly Return Heatmap of {hm_name}")
            st.pyplot(fig)
        else:
            st.error(f"Stock '{hm_name}' not found.")

#"""===================================SIDEBAR AI==================================="""
with st.sidebar:
    st.header("🤖 AI Technical Analyst")
    st.markdown("เลือกเงื่อนไขเพื่อค้นหาหุ้นที่น่าสนใจ")

    # --- Dynamic Filters ---
    market_options = ["All", "SET", "mai"]
    selected_market = st.selectbox("1. เลือกตลาด", market_options)

    # Filter sectors based on selected market
    if selected_market == "All":
        filtered_sectors_df = raw_std
    else:
        filtered_sectors_df = raw_std[raw_std['market'] == selected_market]
    
    sector_options = ["All"] + sorted(filtered_sectors_df['sector'].unique().tolist())
    selected_sector = st.selectbox("2. เลือก Sector", sector_options)

    # Filter sub-sectors based on selected sector
    if selected_sector == "All":
        filtered_subsectors_df = filtered_sectors_df
    else:
        filtered_subsectors_df = filtered_sectors_df[filtered_sectors_df['sector'] == selected_sector]

    sub_sector_options = ["All"] + sorted(filtered_subsectors_df['sub-sector'].dropna().unique().tolist())
    selected_sub_sector = st.selectbox("3. เลือก Sub-sector", sub_sector_options)

    # Analysis Button
    if st.button("วิเคราะห์หุ้นที่น่าสนใจ"):
        analysis_prompt = f"""
            **Persona:** You are a seasoned Technical Analyst providing a clear, actionable summary for an investor. Your analysis must be grounded *exclusively* in the provided data. Your tone should be insightful, objective, and easy to understand.

            **Task:** Based on the filtered data for **Market: '{selected_market}', Sector: '{selected_sector}', Sub-sector: '{selected_sub_sector}'**, identify the most technically outstanding stocks and explain your reasoning.

            **Data Dictionary for Analysis:**
            - **RS Score & Rank:** Key momentum indicators. A high RS Score (closer to 1) and a low Rank (e.g., < 200) signify strong market outperformance.
            - **Price/Volume %Change (D, W):** Short-term price action and confirmation. High positive values suggest strong buying interest.
            - **52WHL:** Position in the 52-week range. A value > 0.8 indicates strength near yearly highs.
            - **ROC_10:** Short-term (10-day) Rate of Change. Positive value confirms upward momentum.

            **Required Output Format (Strictly follow this structure, using Markdown):**

            #### 📈 บทวิเคราะห์ทางเทคนิค ({datetime.date.today().strftime('%d-%m-%Y')})
            **เงื่อนไข:** ตลาด `{selected_market}` | Sector `{selected_sector}` | Sub-sector `{selected_sub_sector}`
            ---

            Based on the data, here are the most technically outstanding stocks:

            **1. 🥇 Most Promising (หุ้นโดดเด่นที่สุด): `[Stock Name]`**
            * **Why it stands out:** This stock shows the best overall combination of strong momentum and positive short-term signals.
            * **RS Score:** `[Value]` (Rank: `[Value]`)
            * **Price Action:** Currently trading near its 52-week high (`52WHL` = `[Value]`) with a strong 10-day momentum (`ROC_10` = `[Value]`).
            * **Volume Confirmation:** Daily volume change is `[Value of Volume %Change D]%`, confirming buying interest.

            **2. 🥈 Strong Momentum (หุ้นโมเมนตัมแข็งแกร่ง): `[Stock Name]`**
            * **Why it stands out:** This stock is a clear market outperformer based on its excellent RS Score.
            * **RS Score:** `[Value]` (Rank: `[Value]`)
            * **Daily Action:** Price changed by `[Value of Price %Change D]%` today.

            **3. 🥉 Rising Star (หุ้นน่าจับตามอง): `[Stock Name]`**
            * **Why it stands out:** This stock is showing strong signs of a potential breakout or upward trend.
            * **Key Signal:** It's near its 52-week high (`52WHL` = `[Value]`) and has a significant positive daily price change (`Price %Change D` = `[Value]`).

            **Analyst's Summary:**
            * [Provide a 1-2 sentence summary. Briefly mention which factor (e.g., RS Score, proximity to 52WHL) was the most decisive in this selection. Conclude with what an investor should watch for next, e.g., "Investors should watch for continued high volume to confirm the trend for the top selections."].

            **Instructions & Rules:**
            1.  From the filtered data, identify three different stocks that fit the criteria for "Most Promising", "Strong Momentum", and "Rising Star".
            2.  **Most Promising:** Find the stock with the highest RS Score that also has a 52WHL > 0.7 and a positive ROC_10.
            3.  **Strong Momentum:** Find the stock with the second-highest RS Score.
            4.  **Rising Star:** Find the stock (not already selected) with the highest `Price %Change D` that also has a 52WHL > 0.6.
            5.  Fill in the template with the exact data for the selected stocks.
            6.  The summary must be objective and directly reference the data.
            7.  If you cannot find stocks that meet all criteria, state that clearly (e.g., "No stocks met the criteria for 'Rising Star' in this selection.") and fill out the sections you can.
            8.  Your entire response must be in Thai (ภาษาไทย).
        """
        
        # All data sources tuple for the AI function
        data_for_ai_tuple = (
            rs_score_data, VP_change_D_data, VP_change_W_data, VP_change_M_data, VP_change_3M_data,
            source_52SET, source_52mai, perf_source_SET, perf_source_mai,
            SET_Stacked_bar, SET_VWAP, SET_VWAP_cumprod,
            mai_Stacked_bar, mai_VWAP, mai_VWAP_cumprod
        )
        
        # Call the AI function with filters and display the response
        response_text = get_ai_response(
            analysis_prompt, 
            data_for_ai_tuple,
            market_filter=selected_market,
            sector_filter=selected_sector,
            sub_sector_filter=selected_sub_sector
        )
        if response_text:
            st.markdown(response_text)

