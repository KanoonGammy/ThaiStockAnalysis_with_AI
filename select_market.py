import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

def select_market(key):
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        first_pick = st.selectbox("Select Market", options= ["All","SET","mai"], key = f"{key} select1")
        sl1,sl2,sl3,sl4 = [None,None,None,None]
    with c2:
        if first_pick in ["All", "SET", "mai"]:
            raw = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET_Index.csv", index_col= "Unnamed: 0")
            raw = raw.sort_values(["symbol"], ascending= True)
            sl1 = first_pick
            if first_pick == "SET":
                select_option = st.selectbox("Select Options",options = ["All", "SET50", "SET100"], key = f"{key} select2")
                if select_option == "All":
                    pass
                elif select_option == "SET50":
                    ticker50 = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET50.csv")['symbol'].tolist()
                    raw = raw[raw["symbol"].isin(ticker50)]
                    sl2 = select_option
                else :
                    ticker100 = pd.read_csv("https://raw.githubusercontent.com/KanoonGammy/Technical-Analysis-Project/refs/heads/main/SET100.csv")['symbol'].tolist()
                    raw = raw[raw["symbol"].isin(ticker100)]
                    sl2 = select_option
            if first_pick == "All":
                pass
            else :
                raw = raw[raw["market"] == f"{first_pick}"]
    with c3:
        select_sector= st.selectbox("Select Sector",options = ["All", *sorted(raw.sector.unique())], key = f"{key} select3")
        sl3 = select_sector
        if select_sector == "All":
            pass
        else :
            raw = raw[raw["sector"] == f"{select_sector}"]
    with c4:
        if first_pick == "SET" :
            select_sub = st.selectbox("Select Sub-Sector", options= ["All", *sorted(raw['sub-sector'].dropna().unique())], key = key)
            
            if select_sub == "All":
                pass
            else:
                raw = raw[raw["sub-sector"] == f"{select_sub}"]
                sl4 = select_sub
        return raw, sl1, sl2, sl3, sl4