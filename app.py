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

# --- Ensure px is a 1-D Series named 'price' ---
if isinstance(df, pd.DataFrame) and "price" in df.columns:
    px = df["price"]
elif isinstance(df, pd.Series):
    px = df.rename("price")
else:
    # last resort: squeeze a single-column frame
    px = pd.Series(df.squeeze(), index=df.index, name="price")

ret = px.pct_change()

# --- Momentum signal (force 1-D) ---
signal = (px.pct_change(lookback) - px.pct_change(skip_recent))
signal = signal.astype(float)                         # ensure numeric
signal = signal.reindex(px.index)                    # align to price index

# --- Positions (force 1-D NumPy -> 1-D Series) ---
sig_vals = signal.values.ravel()                     # <-- guarantees 1-D
pos_vals = np.where(sig_vals > 0, 1.0, (-1.0 if allow_short else 0.0))
pos = pd.Series(pos_vals, index=signal.index, name="pos")

# align to px index, then lag to avoid look-ahead
pos = pos.reindex(px.index, fill_value=0.0).shift(1).fillna(0.0)

# --- Turnover & simple transaction costs ---
turnover = pos.diff().abs().fillna(0.0)
tc = (tc_bps / 10000.0) * turnover

# --- Strategy returns (net of simple costs) ---
strat = pos * ret - tc


turnover = pos.diff().abs().fillna(0)
tc = (tc_bps / 10000.0) * turnover   # simple cost per position change
strat = pos * ret - tc

# -------- Helper: perf stats --------
def perf_stats(returns, freq=252):
    # Coerce to a 1-D Series
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] == 1:
            r = returns.iloc[:, 0].dropna()
        else:
            # If a DataFrame slipped in, take equal-weighted mean across cols
            r = returns.mean(axis=1).dropna()
    elif isinstance(returns, pd.Series):
        r = returns.dropna()
    else:
        r = pd.Series(returns).dropna()

    n = len(r)
    if n < 2:
        return {"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "Max Drawdown": np.nan, "Hit Rate": np.nan}

    cagr = (1 + r).prod() ** (freq / n) - 1
    vol = float(r.std() * np.sqrt(freq))  # ensure scalar float
    sharpe = (cagr / vol) if (np.isfinite(vol) and vol > 0) else np.nan

    cum = (1 + r).cumprod()
    maxdd = (cum / cum.cummax() - 1).min()
    hit = (r > 0).mean()

    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Max Drawdown": maxdd, "Hit Rate": hit}


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
