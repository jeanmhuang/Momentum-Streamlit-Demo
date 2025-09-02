import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Momentum Backtest (Demo)", layout="wide")

# ---------------- Sidebar controls ----------------
st.sidebar.title("Momentum Backtest")
ticker = st.sidebar.text_input("Ticker", "SPY").upper()
start = st.sidebar.date_input("Start date", dt.date(2015, 1, 1))
lookback = st.sidebar.slider("Lookback (trading days ~12m=252)", 60, 504, 252, step=21)
skip_recent = st.sidebar.slider("Exclude recent (days ~1m=21)", 0, 60, 21, step=7)
allow_short = st.sidebar.checkbox("Allow short when signal ≤ 0", True)
tc_bps = st.sidebar.number_input("Transaction cost (bps per side)", 0.0, 100.0, 10.0, step=1.0)

# ---------------- Data loader ----------------
@st.cache_data(show_spinner=False)
def load_prices(ticker: str, start_date: dt.date) -> pd.Series:
    """Return 1-D price series (float) indexed by date."""
    df = yf.download(ticker, start=start_date, auto_adjust=True, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    px = df.get("Close", pd.Series(dtype=float)).astype(float)
    px.name = "price"
    return px.dropna()

# ---------------- Perf stats ----------------
def perf_stats(returns: pd.Series, freq: int = 252) -> dict:
    """Robust stats on a 1-D return series."""
    if isinstance(returns, pd.DataFrame):
        r = returns.mean(axis=1).astype(float).dropna()
    else:
        r = pd.Series(returns).astype(float).dropna()

    n = len(r)
    if n < 2:
        nan = np.nan
        return {"CAGR": nan, "Vol": nan, "Sharpe": nan, "Max Drawdown": nan, "Hit Rate": nan}

    cagr = (1.0 + r).prod() ** (freq / n) - 1.0
    vol = float(r.std() * np.sqrt(freq))
    sharpe = (cagr / vol) if (np.isfinite(vol) and vol > 0) else np.nan
    cum = (1.0 + r).cumprod()
    maxdd = (cum / cum.cummax() - 1.0).min()
    hit = (r > 0).mean()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Max Drawdown": maxdd, "Hit Rate": hit}

# ---------------- Download data ----------------
with st.spinner("Downloading prices…"):
    px = load_prices(ticker, start)

if px.empty:
    st.error("No data returned. Try another ticker or an earlier start date.")
    st.stop()

# force 1-D series (paranoia)
px = pd.Series(px.values, index=px.index, name="price").astype(float)

# Guard: enough history for selected windows
min_len = max(lookback, skip_recent) + 5
if len(px) < min_len:
    st.error("Not enough price history for selected parameters. Choose an earlier start date or smaller lookback.")
    st.stop()

# ---------------- Signal / positions / returns ----------------
# Returns
ret = pd.Series(px.pct_change().values, index=px.index, name="ret").astype(float)

# Momentum signal: past lookback minus recent window (12–1 proxy)
sig_raw = (px.pct_change(lookback) - px.pct_change(skip_recent))
signal = pd.Series(sig_raw.values, index=px.index, name="signal").astype(float)

# Positions from signal (1 for long, -1 or 0 otherwise), lag 1 day to avoid look-ahead
pos_vals = np.where(signal.values > 0.0, 1.0, (-1.0 if allow_short else 0.0))
pos = pd.Series(pos_vals, index=px.index, name="pos").shift(1).fillna(0.0)

# Turnover & simple transaction costs (cost applied when position changes)
turnover = pd.Series(pos.diff().abs().fillna(0.0).values, index=px.index, name="turnover")
tc = pd.Series((tc_bps / 10000.0) * turnover.values, index=px.index, name="tc")

# Strategy net returns
strat = pd.Series((pos * ret - tc).values, index=px.index, name="strat").astype(float)

# ---------------- Stats ----------------
bh_stats = perf_stats(ret)
st_stats = perf_stats(strat)

# ---------------- Layout ----------------
left, right = st.columns([3, 2])

with left:
    st.title("Momentum Strategy vs Buy & Hold")
    st.caption("Signal = past lookback return minus recent return (12–1 proxy). Positions are lagged by 1 day. Simple turnover-based costs.")

    # Align & build equity curves safely
    idx = ret.index.intersection(strat.index)
    ret_series = ret.loc[idx].dropna()
    strat_series = strat.loc[idx].dropna()

    if ret_series.empty or strat_series.empty or len(idx) == 0:
        st.error("No overlapping data after alignment. Adjust parameters.")
        st.stop()

    buyhold_curve = (1.0 + ret_series).cumprod()
    strategy_curve = (1.0 + strat_series).cumprod()

    eq = pd.DataFrame({"Buy & Hold": buyhold_curve, "Strategy (Net)": strategy_curve}, index=idx).dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    eq.plot(ax=ax)
    ax.set_title(f"Growth of $1 — {ticker}")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Drawdown (Strategy)")
    dd = strategy_curve / strategy_curve.cummax() - 1.0
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
        "Metric": ["CAGR", "Vol", "Sharpe", "Max Drawdown", "Hit Rate"],
        "Buy & Hold": [bh_stats[k] for k in ["CAGR","Vol","Sharpe","Max Drawdown","Hit Rate"]],
        "Strategy (Net)": [st_stats[k] for k in ["CAGR","Vol","Sharpe","Max Drawdown","Hit Rate"]],
    }).set_index("Metric").applymap(lambda x: f"{x:.2%}" if isinstance(x, float) and not np.isnan(x) else "—")
    st.dataframe(perf)

st.divider()
st.caption("Educational demo only; simplified assumptions; not investment advice.")
