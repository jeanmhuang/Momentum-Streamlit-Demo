import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Momentum Backtest (Demo)", layout="wide")

# -------- Sidebar controls --------
st.sidebar.title("Momentum Backtest")
ticker = st.sidebar.text_input("Ticker", "SPY").upper()
start = st.sidebar.date_input("Start date", dt.date(2015, 1, 1))
lookback = st.sidebar.slider("Lookback (trading days ~12m=252)", 60, 504, 252, step=21)
skip_recent = st.sidebar.slider("Exclude recent (days ~1m=21)", 0, 60, 21, step=7)
allow_short = st.sidebar.checkbox("Allow short when signal ≤ 0", True)
tc_bps = st.sidebar.number_input("Transaction cost (bps per side)", 0.0, 100.0, 10.0, step=1.0)

# -------- Data --------
@st.cache_data(show_spinner=False)
def load_prices(ticker, start_date):
    df = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
    if df.empty:
        return pd.DataFrame()
    return df[["Close"]].rename(columns={"Close": "price"}).dropna()

with st.spinner("Downloading prices…"):
    df = load_prices(ticker, start)

if df.empty:
    st.error("No data returned. Try another ticker or date.")
    st.stop()

px = df["price"]
ret = px.pct_change()

# -------- Signal & positions (same logic as your Colab cell) --------
signal = px.pct_change(lookback) - px.pct_change(skip_recent)
pos = np.where(signal > 0, 1, (-1 if allow_short else 0))
pos = pd.Series(pos, index=px.index).shift(1).fillna(0)

turnover = pos.diff().abs().fillna(0)
tc = (tc_bps / 10000.0) * turnover   # simple cost per position change
strat = pos * ret - tc

# -------- Helper: perf stats --------
def perf_stats(returns, freq=252):
    r = returns.dropna()
    if r.empty: 
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan, "Hit": np.nan}
    cagr = (1 + r).prod() ** (freq / len(r)) - 1
    vol = r.std() * np.sqrt(freq)
    sharpe = cagr / vol if vol else np.nan
    cum = (1 + r).cumprod()
    maxdd = (cum / cum.cummax() - 1).min()
    hit = (r > 0).mean()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": maxdd, "Hit": hit}

bh_stats = perf_stats(ret)
st_stats = perf_stats(strat)

# -------- Layout --------
left, right = st.columns([3, 2])

with left:
    st.title("Momentum Strategy vs Buy & Hold")
    st.caption("Signal = past lookback return minus recent return (12–1 proxy). Positions lagged by 1 day. Simple turnover-based costs.")

    eq = pd.DataFrame({
        "Buy & Hold": (1 + ret).cumprod(),
        "Strategy (Net)": (1 + strat).cumprod()
    }).dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    eq.plot(ax=ax)
    ax.set_title(f"Growth of $1 — {ticker}")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Drawdown (Strategy)")
    cum = (1 + strat).cumprod()
    dd = cum / cum.cummax() - 1
    fig2, ax2 = plt.subplots(figsize=(10, 2.8))
    dd.plot(ax=ax2)
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2, clear_figure=True)

with right:
    st.subheader("Settings")
    st.markdown(f"- **Ticker**: `{ticker}`")
    st.markdown(f"- **Start**: `{start}`")
    st.markdown(f"- **Lookback**: **{lookback}** days")
    st.markdown(f"- **Exclude recent**: **{skip_recent}** days")
    st.markdown(f"- **Shorting**: {'Enabled' if allow_short else 'Disabled'}")
    st.markdown(f"- **Txn cost**: **{tc_bps:.0f} bps per side**")

    st.subheader("Performance (annualized)")
    perf = pd.DataFrame({
        "Metric": ["CAGR", "Vol", "Sharpe", "MaxDD", "Hit"],
        "Buy & Hold": [bh_stats[k] for k in ["CAGR","Vol","Sharpe","MaxDD","Hit"]],
        "Strategy (Net)": [st_stats[k] for k in ["CAGR","Vol","Sharpe","MaxDD","Hit"]],
    }).set_index("Metric").applymap(lambda x: f"{x:.2%}" if isinstance(x,float) and not np.isnan(x) else "—")
    st.dataframe(perf)

st.divider()
st.caption("Educational demo only; simplified assumptions; not investment advice.")
