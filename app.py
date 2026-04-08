import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import date, timedelta

st.set_page_config(page_title="Stock Correlation Tool", layout="wide")

st.title("Stock Correlation Tool")
st.markdown("Enter a list of tickers, pick a date range, and explore how they move together.")

# --- Inputs ---
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    raw_tickers = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL, AMZN, META",
        help="e.g. AAPL, MSFT, TSLA"
    )

with col2:
    start_date = st.date_input("Start date", value=date.today() - timedelta(days=365))

with col3:
    end_date = st.date_input("End date", value=date.today())

tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

# --- Fetch & compute ---
if st.button("Run", type="primary") or True:
    if len(tickers) < 2:
        st.warning("Please enter at least 2 tickers.")
        st.stop()

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    with st.spinner("Fetching price data..."):
        raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)

    # Handle single vs multi-ticker response shape
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    # Drop tickers with no data
    prices = prices.dropna(axis=1, how="all")
    missing = set(tickers) - set(prices.columns)
    if missing:
        st.warning(f"No data found for: {', '.join(sorted(missing))}")

    if prices.shape[1] < 2:
        st.error("Need at least 2 tickers with valid data.")
        st.stop()

    returns = prices.pct_change().dropna()
    corr = returns.corr()

    # --- Heatmap ---
    st.subheader("Correlation Matrix")
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        labels={"color": "Correlation"},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # --- Raw correlation table ---
    with st.expander("Raw correlation table"):
        st.dataframe(corr.round(4))

    # --- Price chart ---
    with st.expander("Normalized price chart (base = 100)"):
        normalized = (prices / prices.iloc[0]) * 100
        fig2 = px.line(normalized, labels={"value": "Indexed price", "variable": "Ticker"})
        fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)
