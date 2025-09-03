import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Momentum Backtest (Demo)", layout="wide")

# ---------------- Sidebar controls ----------------
st.sidebar.title("Momentum Backtest")
# enforce single ticker just in case user pastes "SPY, AAPL"
ticker = st.sidebar.text_input("Ticker", "SPY").split(",")[0].strip().upper()
start = st.sidebar.date_input("Start date", dt.date(2015, 1, 1))
lookback = st.sidebar.slider("Lookback (trading days ~12m=252)", 60, 504, 252, step=21)
skip_recent = st.sidebar.slider("Exclude recent (days ~1m=21)", 0, 60, 21, step=7)
allow_short = st.sidebar.checkbox("Allow short when signal ≤ 0", True)
tc_bps = st.sidebar.number_input("Transaction cost (bps per side)", 0.0, 100.0, 10.0, step=1.0)

# ---------------- Data loader (always returns 1-D Series) ----------------
@st.cache_data(show_spinner=False)
def load_price_series(t: str, start_date: dt.date) -> pd.Series:
    """
    Download daily prices and return a 1-D float Series named 'price', robust to MultiIndex columns.
    """
    df = yf.download(t, start=start_date, auto_adjust=True, progress=False)
    if df.empty:
        return pd.Series(dtype=float)

    # If yfinance returns MultiIndex columns (e.g., ('Close','SPY'))
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", t) in df.columns:
            px = df[("Close", t)]
        elif ("Adj Close", t) in df.columns:
            px = df[("Adj Close", t)]
        else:
            # fallback: first column with top level Close / Adj Close
            px = None
            for col in df.columns:
                top = col[0] if isinstance(col, tuple) else col
                if top in ("Close", "Adj Close"):
                    px = df[col]
                    break
            if px is None:
                return pd.Series(dtype=float)
    else:
        # Single-index columns
        if "Close" in df.columns:
            px = df["Close"]
        elif "Adj Close" in df.columns:
            px = df["Adj Close"]
        else:
            return pd.Series(dtype=float)

    # Ensure 1-D Series with float dtype and clean name
    px = pd.Series(px, index=df.index, name="price").astype(float).dropna()
    return px

# ---------------- Perf stats (robust) ----------------
def perf_stats(returns: pd.Series, freq: int = 252) -> dict:
    """
    Compute basic performance stats on a 1-D return series.
    """
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

# ---------------- Download & basic guards ----------------
with st.spinner("Downloading prices…"):
    px = load_price_series(ticker, start)

if px.empty:
    st.error("No data returned. Try another ticker or an earlier start date.")
    st.stop()

# Guard: enough history for selected windows
min_len = max(lookback, skip_recent) + 5
if len(px) < min_len:
    st.error("Not enough price history for selected parameters. Choose an earlier start date or smaller lookback.")
    st.stop()

# ---------------- Signal / positions / returns (all forced to 1-D Series) ----------------
# Returns
ret = px.pct_change().astype(float)
ret.name = "ret"

# Momentum signal: past lookback minus recent window (12–1 proxy)
signal = (px.pct_change(lookback) - px.pct_change(skip_recent)).astype(float)
signal.name = "signal"

# Positions from signal (1 for long, -1 or 0 otherwise), then lag 1 day to avoid look-ahead
pos = pd.Series(np.where(signal.values > 0.0, 1.0, (-1.0 if allow_short else 0.0)),
                index=px.index, name="pos").shift(1).fillna(0.0)

# Turnover & simple transaction costs (cost applied when position changes)
turnover = pos.diff().abs().fillna(0.0)
tc = (tc_bps / 10000.0) * turnover

# Strategy net returns
strat = (pos * ret - tc).astype(float)
strat.name = "strat"

# ---------------- Stats ----------------
bh_stats = perf_stats(ret)
st_stats = perf_stats(strat)

# ---------------- Layout ----------------
left, right = st.columns([3, 2])

with left:
    st.title("Momentum Strategy vs Buy & Hold")
    st.caption("Signal = past lookback return minus recent return (12–1 proxy). Positions lagged by 1 day. Simple turnover-based costs.")

    # Align both series and build equity curves
    idx = ret.index.intersection(strat.index)
    ret_series = ret.loc[idx].dropna()
    strat_series = strat.loc[idx].dropna()

    if ret_series.empty or strat_series.empty:
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

