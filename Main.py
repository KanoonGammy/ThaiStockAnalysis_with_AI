import pandas as pd
import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
 
st.set_page_config(layout="wide")

st.header("APG Station")
from select_market import select_market

col1, col2 = st.columns(2)
with col1:
    raw, sl1,sl2,sl3,sl4 = select_market(key = "search main")
with col2:
    ticker_search = str.upper(st.text_input("Search: ", value=''))
raw.reset_index(inplace= True)
raw.index = raw.index + 1
raw = raw.drop("index", axis = 1)
if ticker_search == '':
    st.table(raw)
elif ticker_search:
    st.table(raw[raw["symbol"] == ticker_search])
else:
    st.table(raw)