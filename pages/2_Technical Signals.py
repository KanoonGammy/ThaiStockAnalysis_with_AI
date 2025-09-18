# import necessary libraries
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_ta as ta
import seaborn as sns
import datetime
import vectorbt as vbt
import matplotlib.pyplot as plt
import time
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import importlib.util
from select_market import select_market
# st.set_page_config(layout="wide")
# from Main import search_menu

st.header("APG Station: Technical Signals")

def import_data(period = 3650):
    # import data price
    dp = pd.read_csv("source_price.csv").tail(period) # from souce
    dp.Date = pd.to_datetime(dp["Date"]) # set Date as Datetime
    dp = dp.set_index("Date") # set index

    # rename columns
    t = [x.split(".")[0] for x in dp.columns] # list comprehesion >> split .BK
    index_set = t.index("^SET") # find ^SET index
    t[index_set] = "SET" # rename ^SET as SET
    dp.columns = t # rename index

    return dp

raw_std = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET_Index.csv", index_col= "Unnamed: 0").sort_values(["symbol"], ascending= True)

SET_tickers = raw_std[raw_std["market"] == "SET"]["symbol"].tolist()
mai_tickers = raw_std[raw_std["market"] == "mai"]["symbol"].tolist()

tab1, tab2 = st.tabs(["CDC Action Zone Signals", "CDC Optimization"])

data_for_CDC = import_data(period = 52)
with tab1:
    st.header("CDC Action Zone Signals Timeframe: Day")

    col1, col2, col3, col4 = st.columns(4)
    @st.cache_data
    def CDC_SET(data, tickers):
        CDC_Signal = pd.DataFrame()

        for tick in tickers:

            data_dict = {
                        "Close": data[tick],
                        "EMA_12" : ta.ema(data[tick], 12),
                        "EMA_26": ta.ema(data[tick], 26),
                        }
            
            df = pd.DataFrame(data_dict)
            df.loc[(df["EMA_12"] > df["EMA_26"]) & (df["EMA_12"].shift(1) < df["EMA_26"].shift(1)), "Signal" ] = "Buy"
            df.loc[(df["EMA_12"] < df["EMA_26"]) & (df["EMA_12"].shift(1) > df["EMA_26"].shift(1)), "Signal" ] = "Sell"

            CDC_Signal[f"{tick}"] = df.iloc[:,[-1]]

        return CDC_Signal

    CDC_SETdata = CDC_SET(data_for_CDC, SET_tickers)
    CDC_maidata = CDC_SET(data_for_CDC, mai_tickers)
    filtered_CDC_SETdata = CDC_SETdata.iloc[-1,:].rename("CDC Signal")
    filtered_CDC_SETdata = filtered_CDC_SETdata.dropna()
    filtered_CDC_maidata = CDC_maidata.iloc[-1,:].rename("CDC Signal")
    filtered_CDC_maidata = filtered_CDC_maidata.dropna()
    with col1:
        st.subheader("SET CDC Signal Buy")
        CDC_Buy_Only_SET = filtered_CDC_SETdata[filtered_CDC_SETdata.isin(['Buy'])]
        st.write(CDC_Buy_Only_SET)

    with col2:
        st.subheader("SET CDC Signal Sell")
        CDC_Sell_Only_SET = filtered_CDC_SETdata[filtered_CDC_SETdata.isin(['Sell'])]
        st.write(CDC_Sell_Only_SET)

    with col3:
        st.subheader("mai CDC Signal Buy")
        CDC_Buy_Only_mai = filtered_CDC_maidata[filtered_CDC_maidata.isin(['Buy'])]
        st.write(CDC_Buy_Only_mai)

    with col4:
        st.subheader("mai CDC Signal Sell")
        CDC_Sell_Only_mai = filtered_CDC_maidata[filtered_CDC_maidata.isin(['Sell'])]
        st.write(CDC_Sell_Only_mai)
    st.divider()
with tab1:
    tab1c1, tab1c2,tab1c3 = st.columns([2,1,2])
    with tab1c1:

        ticker = str.upper(st.text_input("ระบุชื่อหุ้นที่ต้องการ","ADVANC"))
        title = st.title(f"Portfolio Performance for {ticker} \nInitial Cash: 10,000 THB")
        data_for_optimization = import_data()
        if ticker in data_for_optimization.columns:

            data_drop = data_for_optimization[ticker].dropna()
            df_dict = {
                    "Close": data_drop,
                    "EMA_short": ta.ema(data_drop,12),
                    "EMA_long": ta.ema(data_drop,26),
                    }
            data_CDCOpt = pd.DataFrame(df_dict)
            data_CDCOpt['Trend'] = data_CDCOpt["EMA_short"] > data_CDCOpt["EMA_long"]
            data_CDCOpt.loc[(data_CDCOpt["Trend"] == True) & (data_CDCOpt["Trend"].shift(1) == False), "Signal"] = "Buy"
            data_CDCOpt.loc[(data_CDCOpt["Trend"] == False) & (data_CDCOpt["Trend"].shift(1) == True), "Signal"] = "Sell"
            data_CDCOpt["Entry_price"] = data_CDCOpt.Close.shift(1)

            entry = (data_CDCOpt.Signal == "Buy")
            exit = (data_CDCOpt.Signal == "Sell")
            init_cash = 10000
            port = vbt.Portfolio.from_signals(data_CDCOpt.Entry_price ,
                                            entries = entry ,
                                            exits = exit ,
                                            fees = 0.001,
                                            # size = 0.5,
                                            init_cash = init_cash ) #Fee 0.1%
            fig = port.plot()
            dict_port = {"Initial Cash": init_cash,
            "กำไรสุทธิกลยุทธ์ (THB)" : [f"{port.total_profit():,.2f}",],
            "คิดเป็น (%)" : [f"{port.total_return()*100:,.2f}%"],
            "รวม" : [f"{port.total_profit() + init_cash:,.2f}"],

            "กำไรสุทธิ Benchmark (THB)" : [f"{port.total_benchmark_return()*init_cash:,.2f}"],
            "กำไร Benchmark (%)" : [f"{port.total_benchmark_return()*100:.2f}"],
            "รวม Benchmark" : [f"{(port.total_benchmark_return()*init_cash) + init_cash:,.2f}"],

            }
            df_port = pd.DataFrame(dict_port)
            st.plotly_chart(fig)

            with tab1c2:
                st.write(port.stats())

            with tab1c3:
                st.write(port.trades.records_readable)
                
        else:
            if ticker != "":
                st.error("ไม่พบหุ้นที่คุณกรอกในข้อมูล")
            else:
                pass



with tab2:
    tab2c1, tab2c2 = st.columns(2)
    price = import_data(3650)
    with tab2c1:
        ticker_opt = str.upper(st.text_input("ระบุชื่อหุ้นที่ต้องการ","ADVANC", key = "Optimization"))
        if ticker_opt :
            st.success(f"กำลังวิเคราะห์หุ้น: {ticker_opt}")

            if ticker_opt in price.columns:
                price = price[ticker_opt]
                windows = np.arange(2, 101,2)
                fast_ma, slow_ma = vbt.MA.run_combs(price, window=windows, r=2, short_names=['fast', 'slow'])
                entries = fast_ma.ma_crossed_above(slow_ma)
                exits = fast_ma.ma_crossed_below(slow_ma)

                pf_kwargs = dict(size=np.inf, fees=0.001, freq='1D')
                pf = vbt.Portfolio.from_signals(price, entries, exits, **pf_kwargs, init_cash= 10000)

                fig = pf.total_return().vbt.heatmap(
                    x_level='fast_window', y_level='slow_window',# slider_level='symbol',
                    symmetric=True,
                    trace_kwargs=dict(colorbar=dict(title='Total return', tickformat='%')))
                st.plotly_chart(fig)
            else:
                if ticker_opt != "":
                    st.error("ไม่พบหุ้นที่คุณกรอกในข้อมูล")
                else:
                    pass
    with tab2c2:
        tab2c2c1, tab2c2c2 = st.columns(2)
        short_ma = int(st.text_input("ระบุเส้นค่าเฉลี่ยระยะสั้น",2, key = "Optimization Int1"))
        long_ma = int(st.text_input("ระบุเส้นค่าเฉลี่ยระยะยาว",4, key = "Optimization Int2"))
        st.dataframe(pf.xs((short_ma,long_ma), level= ("fast_window", "slow_window")).stats())
        
    