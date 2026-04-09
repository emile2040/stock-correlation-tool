import re
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from scipy.stats import spearmanr

st.set_page_config(page_title="Stock Correlation Tool", layout="wide")

st.title("Stock Correlation Tool")
st.markdown("Enter a list of tickers, pick a date range, and explore how they move together.")

# --- Inputs ---
# Ticker input: tab switcher between manual entry and file upload
ticker_tab1, ticker_tab2 = st.tabs(["Type tickers", "Upload file"])

with ticker_tab1:
    raw_tickers_manual = st.text_area(
        "Tickers (comma-separated or one per line)",
        value="AAPL, MSFT, GOOGL, AMZN, META",
        height=80,
        help="e.g. AAPL, MSFT, TSLA — separate by commas, spaces, or new lines",
    )

with ticker_tab2:
    uploaded_file = st.file_uploader(
        "Upload a .txt or .csv file — one ticker per line, or comma-separated",
        type=["txt", "csv"],
    )
    if uploaded_file:
        raw_tickers_manual = uploaded_file.read().decode("utf-8")
        st.success(f"File loaded: {uploaded_file.name}")

# Parse tickers — if CSV with a 'ticker' column, extract that column only;
# otherwise strip comments (#...) and split on whitespace/commas/semicolons
def parse_tickers(raw, is_csv=False):
    if is_csv:
        try:
            df_tickers = pd.read_csv(pd.io.common.StringIO(raw))
            if "ticker" in df_tickers.columns:
                return [t.strip().upper() for t in df_tickers["ticker"].dropna() if str(t).strip()]
        except Exception:
            pass
    lines = [line.split("#")[0] for line in raw.splitlines()]
    cleaned = " ".join(lines)
    return [t.strip().upper() for t in re.split(r"[\s,;]+", cleaned) if t.strip()]

is_csv = uploaded_file is not None and uploaded_file.name.endswith(".csv")
tickers = parse_tickers(raw_tickers_manual, is_csv=is_csv)
if tickers:
    st.caption(f"{len(tickers)} ticker(s): {', '.join(tickers)}")


col2, col3, col4, col5, col6 = st.columns(5)

with col2:
    start_date = st.date_input("Start date", value=date.today() - timedelta(days=3*365))

with col3:
    end_date = st.date_input("End date", value=date.today())

with col4:
    frequency = st.radio(
        "Return frequency",
        options=["Daily", "Weekly"],
        index=1,
        help="Daily: one observation per trading day. Weekly: Mon-to-Mon, Tue-to-Tue, … averaged across the 5 weekday series.",
    )

with col5:
    corr_method = st.radio(
        "Correlation method",
        options=["Spearman", "Pearson"],
        index=0,
        help="Spearman: rank-based, robust to outliers. Pearson: linear, assumes normally distributed returns.",
    )

with col6:
    lam = st.number_input(
        "Lambda",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Adjusts correlations upward: new = corr + (1 − corr) × λ. At 0 = no adjustment, at 1 = all correlations become 1.",
    )

# --- Fetch & compute ---
if st.button("Run", type="primary") or True:
    if len(tickers) < 2:
        st.warning("Please enter at least 2 tickers.")
        st.stop()

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    def download_prices(ticker_list, start, end):
        """Download prices and return a Close price DataFrame."""
        raw = yf.download(ticker_list, start=start, end=end, auto_adjust=True, repair=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            return raw["Close"]
        close = raw[["Close"]]
        close.columns = ticker_list
        return close

    with st.spinner(f"Fetching price data for {len(tickers)} tickers..."):
        prices = download_prices(tickers, start_date, end_date)

    # Drop tickers with no data
    prices = prices.dropna(axis=1, how="all")
    missing = set(tickers) - set(prices.columns)
    if missing:
        with st.expander(f"⚠️ {len(missing)} ticker(s) returned no data — click to see list"):
            st.write(", ".join(sorted(missing)))

    if prices.shape[1] < 2:
        st.error("Need at least 2 tickers with valid data.")
        st.stop()

    method = corr_method.lower()

    if frequency == "Daily":
        returns = prices.pct_change().dropna(how="any")
        corr = returns.corr(method=method)
        caption = f"Based on daily total-return prices ({len(returns)} observations) · {corr_method} correlation · λ={lam}. Adjusted for dividends, splits, and all corporate actions."
    else:
        DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        corr_matrices = []
        obs_counts = []

        for weekday in range(5):
            day_prices = prices[prices.index.dayofweek == weekday]
            day_returns = day_prices.pct_change().dropna(how="any")
            if len(day_returns) >= 2:
                corr_matrices.append(day_returns.corr(method=method))
                obs_counts.append((DAY_NAMES[weekday], len(day_returns)))

        if not corr_matrices:
            st.error("Not enough data to compute weekly correlations. Try a longer date range.")
            st.stop()

        corr = sum(corr_matrices) / len(corr_matrices)
        days_used = ", ".join(f"{d} ({n} obs)" for d, n in obs_counts)
        caption = (
            f"Based on weekly total-return prices — averaged across {len(corr_matrices)} weekday series "
            f"({days_used}) · {corr_method} correlation · λ={lam}. Adjusted for dividends, splits, and all corporate actions."
        )

    # --- Lambda adjustment: corr + (1 - corr) * lambda ---
    if lam > 0:
        corr = corr + (1 - corr) * lam

    # --- Heatmap ---
    st.subheader("Correlation Matrix")
    st.caption(caption)
    n = len(corr)
    cell_size = 50 if n <= 15 else 30 if n <= 30 else 18
    fig_height = max(400, n * cell_size)
    fig = px.imshow(
        corr,
        text_auto=".2f" if n <= 20 else False,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        labels={"color": "Correlation"},
    )
    fig.update_layout(
        height=fig_height,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Average correlation (off-diagonal elements only) ---
    mask = ~np.eye(len(corr), dtype=bool)
    off_diag_values = corr.values[mask]
    avg_corr = off_diag_values.mean()
    min_corr = off_diag_values.min()
    max_corr = off_diag_values.max()

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Average pairwise correlation", f"{avg_corr:.4f}")
    mcol2.metric("Min pairwise correlation", f"{min_corr:.4f}")
    mcol3.metric("Max pairwise correlation", f"{max_corr:.4f}")

    # --- Top 3 lowest / highest average correlation per stock ---
    # For each stock, compute its average correlation with all other stocks
    avg_per_stock = corr.apply(lambda col: col[col.index != col.name].mean())
    lowest3 = avg_per_stock.nsmallest(3)
    highest3 = avg_per_stock.nlargest(3)

    lcol, rcol = st.columns(2)
    with lcol:
        st.markdown("**Lowest avg correlation with universe**")
        for ticker, val in lowest3.items():
            st.markdown(f"- {ticker}: `{val:.4f}`")
    with rcol:
        st.markdown("**Highest avg correlation with universe**")
        for ticker, val in highest3.items():
            st.markdown(f"- {ticker}: `{val:.4f}`")

    # --- Raw correlation table ---
    with st.expander("Raw correlation table"):
        st.dataframe(corr.round(4))

    # --- Price chart ---
    with st.expander("Normalized price chart (base = 100)"):
        normalized = (prices / prices.iloc[0]) * 100
        fig2 = px.line(normalized, labels={"value": "Indexed price", "variable": "Ticker"})
        fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    # ------------------------------------------------------------------ #
    # --- Rolling historical correlation --------------------------------- #
    # ------------------------------------------------------------------ #
    st.divider()
    st.subheader("Rolling Historical Correlation")

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        total_months = st.number_input("Total period (months)", min_value=6, max_value=360, value=120, step=6)
    with rcol2:
        window_months = st.number_input("Rolling window (months)", min_value=3, max_value=240, value=36, step=3)

    if window_months >= total_months:
        st.warning("Rolling window must be shorter than the total period.")
    else:
        roll_start = date.today() - timedelta(days=int(total_months * 30.44))

        with st.spinner(f"Fetching {total_months}-month price data..."):
            prices_roll = download_prices(tickers, roll_start, date.today())

        prices_roll = prices_roll.dropna(axis=1, how="all")

        if prices_roll.shape[1] < 2:
            st.warning("Not enough valid tickers for rolling correlation.")
        else:
            def compute_rolling_avg(returns_df, window_obs, m):
                """Return a Series of average off-diagonal correlation per date."""
                n = returns_df.shape[1]
                off_diag = ~np.eye(n, dtype=bool)
                if m == "pearson":
                    roll_corr = returns_df.rolling(window=window_obs).corr()
                    return (
                        roll_corr.groupby(level=0)
                        .apply(lambda mat: mat.values[off_diag].mean())
                        .dropna()
                    )
                else:
                    arr, idx = returns_df.values, returns_df.index
                    dates_out, avg_out = [], []
                    for i in range(window_obs - 1, len(arr)):
                        w = arr[i - window_obs + 1: i + 1]
                        corr_mat, _ = spearmanr(w)
                        if n == 2:
                            corr_mat = np.array([[1.0, corr_mat], [corr_mat, 1.0]])
                        avg_out.append(corr_mat[off_diag].mean())
                        dates_out.append(idx[i])
                    return pd.Series(avg_out, index=dates_out)

            # Pre-compute both return series
            daily_returns = prices_roll.pct_change().dropna(how="any")
            weekly_returns = prices_roll.resample("W").last().pct_change().dropna(how="any")
            window_obs_daily  = int(window_months * 21)    # ~21 trading days/month
            window_obs_weekly = int(window_months * 4.33)  # ~4.33 weeks/month

            COMBOS = [
                ("Daily · Spearman",  daily_returns,  window_obs_daily,  "spearman"),
                ("Daily · Pearson",   daily_returns,  window_obs_daily,  "pearson"),
                ("Weekly · Spearman", weekly_returns, window_obs_weekly, "spearman"),
                ("Weekly · Pearson",  weekly_returns, window_obs_weekly, "pearson"),
            ]

            series_dict = {}
            any_short = False
            with st.spinner("Computing rolling correlations for all 4 series..."):
                for label, ret, w_obs, m in COMBOS:
                    if len(ret) < w_obs:
                        any_short = True
                        continue
                    series_dict[label] = compute_rolling_avg(ret, w_obs, m)

            if any_short:
                st.warning("Some series were skipped — not enough data for the selected window. Try a smaller rolling window or longer total period.")

            if series_dict:
                fig_roll = go.Figure()
                for label, s in series_dict.items():
                    fig_roll.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=label))
                fig_roll.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_roll.update_layout(
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis=dict(range=[-1, 1], title="Avg pairwise correlation"),
                    xaxis_title="Date",
                    legend_title="Method",
                )
                st.plotly_chart(fig_roll, use_container_width=True)
                st.caption(
                    f"{window_months}-month rolling average pairwise correlation · all frequency/method combinations · "
                    f"{total_months}-month lookback · adjusted for dividends and corporate actions."
                )
