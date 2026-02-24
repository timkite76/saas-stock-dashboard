import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional

st.set_page_config(page_title="SaaS Stock Dashboard", layout="wide", page_icon="📈")

# ─────────────────────────────────────────────
# COMPANY REGISTRY
# ─────────────────────────────────────────────
SAAS_COMPANIES = {
    "Enterprise Application Software": [
        "ADBE", "CRM", "NOW", "WDAY", "INTU", "ADSK", "ANSS", "TYL", "MANH",
        "GWRE", "PCOR", "DOCU", "ORCL", "SAP", "DBX", "BOX", "PTC", "WK", "AGYS",
    ],
    "Cybersecurity & Identity": [
        "CRWD", "PANW", "ZS", "FTNT", "S", "CYBR", "TENB", "QLYS", "RPD", "VRNS",
        "OKTA", "RBRK", "SAIL", "CLBT", "SWI", "RDWR", "FSLY",
    ],
    "Cloud Infrastructure & DevOps": [
        "DDOG", "NET", "MDB", "SNOW", "CFLT", "ESTC", "PATH", "DT", "GTLB", "MNDY",
        "FROG", "DOCN", "NTNX", "AVPT", "OS", "PD",
    ],
    "Communications & Collaboration": [
        "ZM", "TEAM", "TWLO", "RNG", "ZI", "FIVN", "BAND", "NICE", "BRZE",
        "ASAN", "LPSN", "CXM",
    ],
    "Financial Technology SaaS": [
        "BILL", "PAYC", "PCTY", "TOST", "SQ", "FOUR", "FLYW", "NCNO", "ALKT",
        "VERX", "ZUO", "AFRM", "TTAN",
    ],
    "Marketing, Commerce & CX": [
        "HUBS", "SHOP", "TTD", "SEMR", "SPSC", "BIGC", "VTEX", "CWAN",
        "KVYO", "GLBE", "WIX", "APP", "SPT", "ZETA",
    ],
    "Healthcare & Life Sciences": [
        "VEEV", "DOCS", "CERT", "HIMS", "PHR", "GDRX", "EVH", "SDGR", "TXG",
        "NTRA", "HCAT", "WEAV",
    ],
    "Human Capital Management": [
        "CDAY", "APPF", "PYCR",
    ],
    "Data Analytics & AI": [
        "DOMO", "PLTR", "AI", "ALTR", "FRSH", "SQSP", "IOT",
        "SOUN", "BBAI", "DUOL",
    ],
    "Vertical / Specialized SaaS": [
        "CSGP", "BSY", "QTWO", "DCBO", "BL", "DV", "INTA", "JAMF",
        "OLO", "PRGS", "EVBG",
    ],
}

BVP_INDEX_TICKERS = {
    "ADBE", "AGYS", "ALKT", "APPF", "ASAN", "TEAM", "AVPT", "BILL", "BL", "BRZE",
    "AI", "CLBT", "CWAN", "NET", "CFLT", "CRWD", "DDOG", "DOCN", "DOCU", "DT",
    "ESTC", "FSLY", "FIVN", "FRSH", "GTLB", "HUBS", "INTA", "FROG", "KVYO", "MNDY",
    "MDB", "NCNO", "NTNX", "OKTA", "OS", "PD", "PLTR", "PANW", "PAYC", "PCTY",
    "PCOR", "QTWO", "QLYS", "RNG", "RBRK", "SAIL", "CRM", "IOT", "SEMR", "S",
    "NOW", "TTAN", "SHOP", "SNOW", "CXM", "SPT", "SPSC", "TENB", "TOST", "TWLO",
    "PATH", "VEEV", "WEAV", "WIX", "WDAY", "WK", "ZETA", "ZS",
}

ALL_TICKERS = sorted(set(t for tickers in SAAS_COMPANIES.values() for t in tickers))
TICKER_TO_SECTOR = {}
for sector, tickers in SAAS_COMPANIES.items():
    for t in tickers:
        TICKER_TO_SECTOR[t] = sector


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def format_large_number(num):
    if num is None or (isinstance(num, float) and np.isnan(num)):
        return "N/A"
    num = float(num)
    if abs(num) >= 1e12:
        return f"${num / 1e12:.2f}T"
    if abs(num) >= 1e9:
        return f"${num / 1e9:.2f}B"
    if abs(num) >= 1e6:
        return f"${num / 1e6:.1f}M"
    if abs(num) >= 1e3:
        return f"${num / 1e3:.1f}K"
    return f"${num:.2f}"


def safe_pct(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.1f}%"


def safe_val(val, fmt=".2f", prefix="", suffix=""):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{prefix}{val:{fmt}}{suffix}"


def pct_rank(series):
    """Percentile rank (0-100) for a series, NaN-safe."""
    return series.rank(pct=True, na_option="keep") * 100


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
INFO_FIELDS = [
    "shortName", "sector", "marketCap", "trailingPE", "forwardPE", "pegRatio",
    "priceToSalesTrailing12Months", "enterpriseToRevenue", "totalRevenue",
    "revenueGrowth", "grossMargins", "operatingMargins", "profitMargins",
    "trailingEps", "forwardEps", "fiftyTwoWeekLow", "fiftyTwoWeekHigh",
    "currentPrice", "previousClose", "regularMarketPrice", "beta",
    "dividendYield", "returnOnEquity", "debtToEquity", "freeCashflow",
    "enterpriseValue",
]


def fetch_single_ticker_info(ticker: str) -> Optional[dict]:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            return None
        row = {"ticker": ticker}
        for field in INFO_FIELDS:
            row[field] = info.get(field)
        price = row.get("currentPrice") or row.get("regularMarketPrice")
        row["currentPrice"] = price
        try:
            hist = t.history(period="ytd")
            if hist is not None and len(hist) > 0:
                first_close = hist["Close"].iloc[0]
                if first_close and first_close > 0 and price:
                    row["ytd_return"] = ((price - first_close) / first_close) * 100
                else:
                    row["ytd_return"] = None
            else:
                row["ytd_return"] = None
        except Exception:
            row["ytd_return"] = None
        return row
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all_fundamentals(_ticker_list: tuple) -> tuple:
    results, unavailable = [], []
    progress_bar = st.progress(0, text="Fetching stock data...")
    total = len(_ticker_list)
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_ticker = {executor.submit(fetch_single_ticker_info, t): t for t in _ticker_list}
        done_count = 0
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            done_count += 1
            progress_bar.progress(done_count / total, text=f"Fetching stock data... {done_count}/{total} ({ticker})")
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    unavailable.append(ticker)
            except Exception:
                unavailable.append(ticker)
    progress_bar.empty()
    df = pd.DataFrame(results)
    if len(df) > 0:
        df["sub_sector"] = df["ticker"].map(TICKER_TO_SECTOR)
        df["bvp_index"] = df["ticker"].isin(BVP_INDEX_TICKERS)
        df["rule_of_40"] = np.where(
            df["revenueGrowth"].notna() & df["profitMargins"].notna(),
            df["revenueGrowth"] * 100 + df["profitMargins"] * 100, np.nan,
        )
        # 52-week range position (0-100%)
        df["range_52w_pct"] = np.where(
            df["fiftyTwoWeekHigh"].notna() & df["fiftyTwoWeekLow"].notna() & (df["fiftyTwoWeekHigh"] > df["fiftyTwoWeekLow"]),
            (df["currentPrice"] - df["fiftyTwoWeekLow"]) / (df["fiftyTwoWeekHigh"] - df["fiftyTwoWeekLow"]) * 100,
            np.nan,
        )
        # FCF yield
        df["fcf_yield"] = np.where(
            df["freeCashflow"].notna() & df["marketCap"].notna() & (df["marketCap"] > 0),
            df["freeCashflow"] / df["marketCap"] * 100, np.nan,
        )
    return df, unavailable


@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_history(ticker: str, period: str) -> pd.DataFrame:
    try:
        return yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_multi_price_history(tickers: tuple, period: str) -> pd.DataFrame:
    try:
        return yf.download(list(tickers), period=period, auto_adjust=True, progress=False, group_by="ticker")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_technical_analytics(_ticker_list: tuple) -> pd.DataFrame:
    """Fetch 1Y prices for all tickers, compute multi-period returns, MAs, drawdown."""
    try:
        price_data = yf.download(list(_ticker_list), period="1y", auto_adjust=True, progress=False, group_by="ticker")
    except Exception:
        return pd.DataFrame()
    if price_data is None or len(price_data) == 0:
        return pd.DataFrame()

    rows = []
    single_ticker = not isinstance(price_data.columns, pd.MultiIndex)
    for ticker in _ticker_list:
        try:
            if single_ticker:
                close = price_data["Close"].dropna()
            else:
                if ticker not in price_data.columns.get_level_values(0):
                    continue
                close = price_data[(ticker, "Close")].dropna()
            if len(close) < 5:
                continue
            current = close.iloc[-1]
            row = {"ticker": ticker}
            for label, days in [("1w", 5), ("1m", 21), ("3m", 63), ("6m", 126), ("1y", 252)]:
                if len(close) > days:
                    row[f"return_{label}"] = ((current - close.iloc[-(days + 1)]) / close.iloc[-(days + 1)]) * 100
                else:
                    row[f"return_{label}"] = None
            if len(close) >= 50:
                ma50 = close.rolling(50).mean().iloc[-1]
                row["ma_50"] = round(ma50, 2)
                row["above_ma50"] = current > ma50
            else:
                row["ma_50"], row["above_ma50"] = None, None
            if len(close) >= 200:
                ma200 = close.rolling(200).mean().iloc[-1]
                row["ma_200"] = round(ma200, 2)
                row["above_ma200"] = current > ma200
                ma50_s = close.rolling(50).mean()
                ma200_s = close.rolling(200).mean()
                diff = (ma50_s - ma200_s).dropna()
                if len(diff) >= 2:
                    if diff.iloc[-1] > 0 and diff.iloc[-2] <= 0:
                        row["signal"] = "Golden Cross"
                    elif diff.iloc[-1] < 0 and diff.iloc[-2] >= 0:
                        row["signal"] = "Death Cross"
                    else:
                        row["signal"] = ""
                else:
                    row["signal"] = ""
            else:
                row["ma_200"], row["above_ma200"], row["signal"] = None, None, ""
            # 6-month max drawdown
            dd_close = close.iloc[-min(126, len(close)):]
            running_max = dd_close.cummax()
            drawdowns = (dd_close - running_max) / running_max * 100
            row["max_drawdown_6m"] = round(drawdowns.min(), 2)
            # Momentum score: average percentile of 1m, 3m, 6m returns (computed later)
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_quarterly_financials(ticker: str) -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        stmt = t.quarterly_income_stmt
        if stmt is not None and len(stmt) > 0:
            return stmt
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_earnings_data(ticker: str) -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        ed = t.get_earnings_dates(limit=12)
        if ed is not None and len(ed) > 0:
            return ed
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("SaaS Stock Dashboard")
st.sidebar.markdown("Real-time data from Yahoo Finance")
st.sidebar.markdown("---")

sector_filter = st.sidebar.multiselect(
    "Filter by Sub-Sector",
    options=sorted(SAAS_COMPANIES.keys()),
    default=sorted(SAAS_COMPANIES.keys()),
)

PERIOD_OPTIONS = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
                  "YTD": "ytd", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y", "Max": "max"}

search_query = st.sidebar.text_input("Search by ticker or name", "")

st.sidebar.markdown("---")
if st.sidebar.button("Refresh All Data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.caption(f"Data refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.sidebar.caption(f"Tracking {len(ALL_TICKERS)} tickers")

# Load fundamental data
df, unavailable_tickers = fetch_all_fundamentals(tuple(ALL_TICKERS))

if unavailable_tickers:
    with st.sidebar.expander(f"Unavailable tickers ({len(unavailable_tickers)})"):
        st.write(", ".join(sorted(unavailable_tickers)))

# Load technical data
with st.spinner("Computing technical analytics..."):
    available_tickers = tuple(sorted(df["ticker"].tolist())) if len(df) > 0 else ()
    tech_df = fetch_technical_analytics(available_tickers) if len(available_tickers) > 0 else pd.DataFrame()

# Merge technical data into main df
if len(df) > 0 and len(tech_df) > 0:
    df = df.merge(tech_df, on="ticker", how="left")

# Apply filters
filtered_df = df.copy()
if len(filtered_df) > 0:
    filtered_df = filtered_df[filtered_df["sub_sector"].isin(sector_filter)]
    if search_query:
        sq = search_query.upper()
        filtered_df = filtered_df[
            filtered_df["ticker"].str.contains(sq, na=False)
            | filtered_df["shortName"].astype(str).str.upper().str.contains(sq, na=False)
        ]

st.sidebar.markdown(f"**Showing {len(filtered_df)} of {len(df)} companies**")


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Market Overview", "Scores & Rankings", "Individual Stock",
    "Multi-Stock Comparison", "SaaS Screener", "Momentum & Signals",
])


# ─────────────────────────────────────────────
# TAB 1 — MARKET OVERVIEW (enhanced)
# ─────────────────────────────────────────────
with tab1:
    st.header("SaaS Market Overview")
    if len(filtered_df) == 0:
        st.warning("No data available. Adjust your filters.")
    else:
        # KPI row
        c1, c2, c3, c4 = st.columns(4)
        total_mcap = filtered_df["marketCap"].sum()
        c1.metric("Total Market Cap", format_large_number(total_mcap))
        c2.metric("Median P/E", safe_val(filtered_df["trailingPE"].median(), ".1f"))
        c3.metric("Median YTD Return", safe_pct(filtered_df["ytd_return"].median()))
        med_rev_growth = filtered_df["revenueGrowth"].median()
        c4.metric("Median Rev Growth", safe_pct(med_rev_growth * 100 if pd.notna(med_rev_growth) else None))

        # ── Sector Performance Heatmap ──
        st.subheader("Sector Performance Heatmap")
        return_cols = [c for c in ["return_1w", "return_1m", "return_3m", "return_6m", "ytd_return", "return_1y"] if c in filtered_df.columns]
        if len(return_cols) > 0:
            sector_perf = filtered_df.groupby("sub_sector")[return_cols].median().reset_index()
            col_labels = {"return_1w": "1W", "return_1m": "1M", "return_3m": "3M",
                          "return_6m": "6M", "ytd_return": "YTD", "return_1y": "1Y"}
            heat_data = sector_perf.set_index("sub_sector")[return_cols].rename(columns=col_labels)
            fig_heat = px.imshow(
                heat_data.values, x=list(heat_data.columns), y=list(heat_data.index),
                color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                text_auto=".1f", aspect="auto",
                labels=dict(color="Median Return %"),
            )
            fig_heat.update_layout(height=max(350, 40 * len(heat_data)), margin=dict(t=20, l=200, r=10, b=10))
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Technical data not available for sector heatmap.")

        # ── 52-Week Range Position ──
        st.subheader("52-Week Range Position")
        if "range_52w_pct" in filtered_df.columns:
            range_df = filtered_df[filtered_df["range_52w_pct"].notna()].sort_values("range_52w_pct", ascending=True).copy()
            if len(range_df) > 0:
                range_view = st.radio("Show", ["Bottom 25 (nearest 52W low)", "Top 25 (nearest 52W high)", "All"],
                                       horizontal=True, key="range_view")
                if range_view.startswith("Bottom"):
                    range_plot = range_df.head(25)
                elif range_view.startswith("Top"):
                    range_plot = range_df.tail(25)
                else:
                    range_plot = range_df
                fig_range = go.Figure()
                fig_range.add_trace(go.Bar(
                    y=range_plot["ticker"], x=range_plot["range_52w_pct"], orientation="h",
                    marker=dict(
                        color=range_plot["range_52w_pct"],
                        colorscale=[[0, "#ef4444"], [0.5, "#eab308"], [1, "#22c55e"]],
                        cmin=0, cmax=100,
                    ),
                    text=range_plot["range_52w_pct"].apply(lambda x: f"{x:.0f}%"),
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Position: %{x:.1f}%<extra></extra>",
                ))
                fig_range.update_layout(
                    height=max(400, 22 * len(range_plot)), xaxis_title="Position in 52-Week Range (%)",
                    xaxis=dict(range=[0, 110]), yaxis_title="", margin=dict(t=20, l=10, r=10, b=10),
                )
                st.plotly_chart(fig_range, use_container_width=True)

        # ── Sector Summary Table ──
        st.subheader("Sector Summary")
        sector_agg = filtered_df.groupby("sub_sector").agg(
            Companies=("ticker", "count"), Avg_Market_Cap=("marketCap", "mean"),
            Median_PE=("trailingPE", "median"), Median_YTD=("ytd_return", "median"),
            Median_Rev_Growth=("revenueGrowth", "median"),
        ).reset_index()
        sector_agg.columns = ["Sub-Sector", "Companies", "Avg Market Cap", "Median P/E", "Median YTD %", "Median Rev Growth"]
        sector_agg["Avg Market Cap"] = sector_agg["Avg Market Cap"].apply(format_large_number)
        sector_agg["Median P/E"] = sector_agg["Median P/E"].apply(lambda x: safe_val(x, ".1f"))
        sector_agg["Median YTD %"] = sector_agg["Median YTD %"].apply(safe_pct)
        sector_agg["Median Rev Growth"] = sector_agg["Median Rev Growth"].apply(lambda x: safe_pct(x * 100 if pd.notna(x) else None))
        st.dataframe(sector_agg, use_container_width=True, hide_index=True)

        # ── Market Cap Treemap ──
        st.subheader("Market Cap Treemap")
        treemap_df = filtered_df[filtered_df["marketCap"].notna() & (filtered_df["marketCap"] > 0)].copy()
        if len(treemap_df) > 0:
            treemap_df["ytd_display"] = treemap_df["ytd_return"].fillna(0)
            fig_tree = px.treemap(
                treemap_df, path=["sub_sector", "ticker"], values="marketCap",
                color="ytd_display", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
                hover_data={"marketCap": True, "ytd_return": True, "trailingPE": True},
                title="Market Cap by Sub-Sector (color = YTD Return %)",
            )
            fig_tree.update_layout(height=650, margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(fig_tree, use_container_width=True)

        # ── All Companies Table ──
        st.subheader("All Companies")
        display_df = filtered_df[["ticker", "shortName", "sub_sector", "currentPrice",
                                   "marketCap", "trailingPE", "ytd_return", "totalRevenue", "trailingEps"]].copy()
        display_df.columns = ["Ticker", "Name", "Sub-Sector", "Price", "Market Cap", "P/E", "YTD %", "Revenue", "EPS"]
        display_df = display_df.sort_values("Market Cap", ascending=False).reset_index(drop=True)
        display_df.index = range(1, len(display_df) + 1)
        st.dataframe(
            display_df.style.format({
                "Price": "${:.2f}", "Market Cap": lambda x: format_large_number(x),
                "P/E": lambda x: safe_val(x, ".1f"), "YTD %": lambda x: safe_pct(x),
                "Revenue": lambda x: format_large_number(x), "EPS": lambda x: safe_val(x, ".2f", prefix="$"),
            }),
            use_container_width=True, height=min(600, 35 * len(display_df) + 38),
        )


# ─────────────────────────────────────────────
# TAB 2 — SCORES & RANKINGS
# ─────────────────────────────────────────────
with tab2:
    st.header("Scores & Rankings")
    if len(filtered_df) == 0:
        st.warning("No data available. Adjust your filters.")
    else:
        # ── Composite Score with adjustable weights ──
        st.subheader("Composite Score")
        st.caption("Adjust weights to prioritize what matters to you. Each metric is percentile-ranked (0-100) then weighted.")
        wc1, wc2, wc3, wc4, wc5, wc6 = st.columns(6)
        w_growth = wc1.slider("Rev Growth", 0, 100, 25, 5, key="w_growth")
        w_margin = wc2.slider("Profit Margin", 0, 100, 20, 5, key="w_margin")
        w_r40 = wc3.slider("Rule of 40", 0, 100, 20, 5, key="w_r40")
        w_momentum = wc4.slider("YTD Momentum", 0, 100, 15, 5, key="w_momentum")
        w_valuation = wc5.slider("Valuation (low P/S)", 0, 100, 10, 5, key="w_valuation")
        w_fcf = wc6.slider("FCF Yield", 0, 100, 10, 5, key="w_fcf")
        total_w = w_growth + w_margin + w_r40 + w_momentum + w_valuation + w_fcf

        scored = filtered_df.copy()
        scored["p_growth"] = pct_rank(scored["revenueGrowth"])
        scored["p_margin"] = pct_rank(scored["profitMargins"])
        scored["p_r40"] = pct_rank(scored["rule_of_40"])
        scored["p_momentum"] = pct_rank(scored["ytd_return"])
        # Invert valuation: lower P/S = better = higher percentile
        scored["p_valuation"] = 100 - pct_rank(scored["priceToSalesTrailing12Months"])
        scored["p_fcf"] = pct_rank(scored["fcf_yield"])

        if total_w > 0:
            scored["composite_score"] = (
                scored["p_growth"] * w_growth + scored["p_margin"] * w_margin +
                scored["p_r40"] * w_r40 + scored["p_momentum"] * w_momentum +
                scored["p_valuation"] * w_valuation + scored["p_fcf"] * w_fcf
            ) / total_w
        else:
            scored["composite_score"] = 50

        scored = scored.sort_values("composite_score", ascending=False).reset_index(drop=True)
        scored.index = range(1, len(scored) + 1)

        # Leaderboard
        leader_cols = ["ticker", "shortName", "sub_sector", "composite_score",
                       "p_growth", "p_margin", "p_r40", "p_momentum", "p_valuation", "p_fcf"]
        leader_display = scored[[c for c in leader_cols if c in scored.columns]].copy()
        leader_display.columns = ["Ticker", "Name", "Sub-Sector", "Composite", "Growth", "Margin",
                                   "Rule40", "Momentum", "Valuation", "FCF"][:len(leader_display.columns)]
        st.dataframe(
            leader_display.style.format({
                "Composite": "{:.1f}", "Growth": "{:.0f}", "Margin": "{:.0f}",
                "Rule40": "{:.0f}", "Momentum": "{:.0f}", "Valuation": "{:.0f}", "FCF": "{:.0f}",
            }).background_gradient(subset=["Composite"], cmap="RdYlGn", vmin=0, vmax=100),
            use_container_width=True, height=min(600, 35 * len(leader_display) + 38),
        )

        # ── EV/Revenue vs Revenue Growth Regression ──
        st.subheader("EV/Revenue vs Revenue Growth (Relative Valuation)")
        reg_df = filtered_df[filtered_df["revenueGrowth"].notna() & filtered_df["enterpriseToRevenue"].notna()].copy()
        if len(reg_df) >= 5:
            reg_df["rev_growth_pct"] = reg_df["revenueGrowth"] * 100
            x = reg_df["rev_growth_pct"].values
            y = reg_df["enterpriseToRevenue"].values
            coeffs = np.polyfit(x, y, 1)
            reg_df["expected_ev_rev"] = coeffs[0] * x + coeffs[1]
            reg_df["premium_discount"] = ((reg_df["enterpriseToRevenue"] - reg_df["expected_ev_rev"]) / reg_df["expected_ev_rev"] * 100)
            reg_df["valuation_label"] = np.where(reg_df["premium_discount"] > 0, "Premium", "Discount")

            fig_reg = px.scatter(
                reg_df, x="rev_growth_pct", y="enterpriseToRevenue",
                color="valuation_label", color_discrete_map={"Premium": "#ef4444", "Discount": "#22c55e"},
                hover_data={"ticker": True, "shortName": True, "rev_growth_pct": ":.1f",
                            "enterpriseToRevenue": ":.1f", "premium_discount": ":.1f"},
                labels={"rev_growth_pct": "Revenue Growth %", "enterpriseToRevenue": "EV / Revenue"},
            )
            x_line = np.linspace(x.min(), x.max(), 100)
            fig_reg.add_trace(go.Scatter(
                x=x_line, y=coeffs[0] * x_line + coeffs[1],
                mode="lines", name="Fair Value Line", line=dict(color="gray", dash="dash", width=2),
            ))
            # Label the 5 biggest outliers
            top_outliers = reg_df.nlargest(3, "premium_discount")
            bottom_outliers = reg_df.nsmallest(3, "premium_discount")
            for _, row in pd.concat([top_outliers, bottom_outliers]).iterrows():
                fig_reg.add_annotation(
                    x=row["rev_growth_pct"], y=row["enterpriseToRevenue"],
                    text=row["ticker"], showarrow=True, arrowhead=2, ax=15, ay=-15, font=dict(size=10),
                )
            fig_reg.update_layout(height=550, margin=dict(t=20, l=10, r=10, b=10))
            st.plotly_chart(fig_reg, use_container_width=True)
            st.caption("Stocks above the line trade at a premium to their growth rate; below the line = discount.")
        else:
            st.info("Not enough data for regression analysis.")

        # ── Percentile Rank Table ──
        st.subheader("Percentile Rank Table")
        st.caption("Every value is a percentile rank (0-100) among tracked companies. Higher = better (valuation is inverted).")
        pct_display = scored[["ticker", "shortName", "sub_sector", "p_growth", "p_margin",
                              "p_r40", "p_momentum", "p_valuation", "p_fcf"]].copy()
        pct_display.columns = ["Ticker", "Name", "Sub-Sector", "Rev Growth", "Profit Margin",
                               "Rule of 40", "YTD Momentum", "Valuation", "FCF Yield"]
        st.dataframe(
            pct_display.style.format({
                "Rev Growth": "{:.0f}", "Profit Margin": "{:.0f}", "Rule of 40": "{:.0f}",
                "YTD Momentum": "{:.0f}", "Valuation": "{:.0f}", "FCF Yield": "{:.0f}",
            }).background_gradient(subset=["Rev Growth", "Profit Margin", "Rule of 40",
                                            "YTD Momentum", "Valuation", "FCF Yield"],
                                    cmap="RdYlGn", vmin=0, vmax=100),
            use_container_width=True, height=min(600, 35 * len(pct_display) + 38),
        )


# ─────────────────────────────────────────────
# TAB 3 — INDIVIDUAL STOCK (enhanced)
# ─────────────────────────────────────────────
with tab3:
    st.header("Individual Stock Analysis")
    if len(filtered_df) == 0:
        st.warning("No data available. Adjust your filters.")
    else:
        ticker_list = sorted(filtered_df["ticker"].tolist())
        selected_ticker = st.selectbox("Select a company", ticker_list, key="individual_ticker")
        stock_row = filtered_df[filtered_df["ticker"] == selected_ticker].iloc[0]

        price = stock_row.get("currentPrice")
        prev_close = stock_row.get("previousClose")
        price_delta = None
        if pd.notna(price) and pd.notna(prev_close) and prev_close > 0:
            price_delta = price - prev_close

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Price", f"${price:.2f}" if pd.notna(price) else "N/A",
                   delta=f"${price_delta:.2f}" if price_delta is not None else None)
        k2.metric("Market Cap", format_large_number(stock_row.get("marketCap")))
        k3.metric("P/E (Trailing)", safe_val(stock_row.get("trailingPE"), ".1f"))
        low52, high52 = stock_row.get("fiftyTwoWeekLow"), stock_row.get("fiftyTwoWeekHigh")
        range_str = f"${low52:.0f} - ${high52:.0f}" if pd.notna(low52) and pd.notna(high52) else "N/A"
        k4.metric("52W Range", range_str)
        k5.metric("YTD Return", safe_pct(stock_row.get("ytd_return")))
        k6.metric("Revenue", format_large_number(stock_row.get("totalRevenue")))

        # Price chart with optional drawdown overlay
        st.subheader(f"{selected_ticker} Price Chart")
        pc1, pc2, pc3 = st.columns([2, 2, 1])
        with pc1:
            selected_period = PERIOD_OPTIONS[st.selectbox("Date Range", list(PERIOD_OPTIONS.keys()), index=4, key="individual_period")]
        with pc2:
            chart_type = st.radio("Chart type", ["Line", "Candlestick"], horizontal=True, key="chart_type")
        with pc3:
            show_drawdown = st.checkbox("Show drawdown", value=False, key="show_dd")

        with st.spinner(f"Loading {selected_ticker} price data..."):
            price_data = fetch_price_history(selected_ticker, selected_period)

        if price_data is not None and len(price_data) > 0:
            if isinstance(price_data.columns, pd.MultiIndex):
                price_data = price_data.droplevel(1, axis=1)

            n_rows = 3 if show_drawdown else 2
            row_heights = [0.55, 0.25, 0.20] if show_drawdown else [0.75, 0.25]
            fig = make_subplots(
                rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                row_heights=row_heights, subplot_titles=("", "Volume", "Drawdown %") if show_drawdown else ("", "Volume"),
            )
            if chart_type == "Candlestick" and all(c in price_data.columns for c in ["Open", "High", "Low", "Close"]):
                fig.add_trace(go.Candlestick(
                    x=price_data.index, open=price_data["Open"], high=price_data["High"],
                    low=price_data["Low"], close=price_data["Close"], name="OHLC",
                ), row=1, col=1)
            elif "Close" in price_data.columns:
                fig.add_trace(go.Scatter(
                    x=price_data.index, y=price_data["Close"], mode="lines", name="Close",
                    line=dict(color="#2962FF"),
                ), row=1, col=1)

            if "Volume" in price_data.columns:
                fig.add_trace(go.Bar(
                    x=price_data.index, y=price_data["Volume"], name="Volume",
                    marker_color="rgba(41, 98, 255, 0.3)",
                ), row=2, col=1)

            if show_drawdown and "Close" in price_data.columns:
                close_s = price_data["Close"].dropna()
                running_max = close_s.cummax()
                dd = (close_s - running_max) / running_max * 100
                fig.add_trace(go.Scatter(
                    x=dd.index, y=dd.values, fill="tozeroy", name="Drawdown",
                    line=dict(color="#ef4444", width=1), fillcolor="rgba(239,68,68,0.3)",
                ), row=3, col=1)
                fig.update_yaxes(title_text="Drawdown %", row=3, col=1)

            fig.update_layout(
                height=600 if show_drawdown else 550, xaxis_rangeslider_visible=False,
                showlegend=False, margin=dict(t=20, l=10, r=10, b=10),
            )
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No price data available for this ticker/period.")

        # ── Quarterly Revenue & Margin Trends ──
        st.subheader("Quarterly Revenue & Margin Trends")
        with st.spinner("Loading quarterly data..."):
            q_data = fetch_quarterly_financials(selected_ticker)
        if q_data is not None and len(q_data) > 0:
            q_transposed = q_data.T.sort_index()
            rev_col = next((c for c in q_transposed.columns if "Total Revenue" in str(c) or "TotalRevenue" in str(c)), None)
            gp_col = next((c for c in q_transposed.columns if "Gross Profit" in str(c) or "GrossProfit" in str(c)), None)
            oi_col = next((c for c in q_transposed.columns if "Operating Income" in str(c) or "OperatingIncome" in str(c)), None)
            ni_col = next((c for c in q_transposed.columns if "Net Income" in str(c) or "NetIncome" in str(c)), None)

            if rev_col is not None:
                qfig = make_subplots(specs=[[{"secondary_y": True}]])
                qfig.add_trace(go.Bar(
                    x=q_transposed.index.astype(str), y=q_transposed[rev_col],
                    name="Revenue", marker_color="#2962FF", opacity=0.7,
                ), secondary_y=False)
                if gp_col is not None and rev_col is not None:
                    gross_margin = (q_transposed[gp_col] / q_transposed[rev_col] * 100).round(1)
                    qfig.add_trace(go.Scatter(
                        x=q_transposed.index.astype(str), y=gross_margin,
                        mode="lines+markers", name="Gross Margin %", line=dict(color="#22c55e", width=2),
                    ), secondary_y=True)
                if oi_col is not None and rev_col is not None:
                    op_margin = (q_transposed[oi_col] / q_transposed[rev_col] * 100).round(1)
                    qfig.add_trace(go.Scatter(
                        x=q_transposed.index.astype(str), y=op_margin,
                        mode="lines+markers", name="Operating Margin %", line=dict(color="#f59e0b", width=2),
                    ), secondary_y=True)
                qfig.update_layout(height=400, margin=dict(t=20, l=10, r=10, b=10))
                qfig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
                qfig.update_yaxes(title_text="Margin %", secondary_y=True)
                st.plotly_chart(qfig, use_container_width=True)
            else:
                st.info("Revenue data not found in quarterly financials.")
        else:
            st.info("Quarterly financial data not available for this ticker.")

        # ── Earnings Surprise ──
        st.subheader("Earnings History")
        with st.spinner("Loading earnings data..."):
            earn_data = fetch_earnings_data(selected_ticker)
        if earn_data is not None and len(earn_data) > 0:
            earn_display = earn_data.copy()
            # Filter to rows that have reported EPS (past earnings)
            if "Reported EPS" in earn_display.columns and "EPS Estimate" in earn_display.columns:
                past = earn_display[earn_display["Reported EPS"].notna()].head(8).copy()
                if len(past) > 0:
                    past["Surprise"] = past["Reported EPS"] - past["EPS Estimate"]
                    past["Surprise %"] = np.where(
                        past["EPS Estimate"].notna() & (past["EPS Estimate"] != 0),
                        (past["Surprise"] / past["EPS Estimate"].abs()) * 100, np.nan,
                    )
                    efig = go.Figure()
                    colors = ["#22c55e" if s >= 0 else "#ef4444" for s in past["Surprise"].fillna(0)]
                    efig.add_trace(go.Bar(
                        x=past.index.astype(str), y=past["Surprise %"],
                        marker_color=colors, name="Surprise %",
                        text=past["Surprise %"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else ""),
                        textposition="outside",
                    ))
                    efig.update_layout(height=350, yaxis_title="Earnings Surprise %",
                                       margin=dict(t=20, l=10, r=10, b=10))
                    st.plotly_chart(efig, use_container_width=True)
                    st.dataframe(past[["EPS Estimate", "Reported EPS", "Surprise", "Surprise %"]].style.format({
                        "EPS Estimate": "${:.2f}", "Reported EPS": "${:.2f}",
                        "Surprise": "${:+.2f}", "Surprise %": "{:+.1f}%",
                    }), use_container_width=True)
                else:
                    st.info("No past earnings data with reported EPS.")
            else:
                st.info("Earnings estimate/actual columns not available.")
        else:
            st.info("Earnings data not available for this ticker.")

        # Fundamental details
        with st.expander("Fundamental Details"):
            fc1, fc2, fc3, fc4 = st.columns(4)
            with fc1:
                st.markdown("**Margins**")
                st.write(f"Gross: {safe_pct(stock_row.get('grossMargins', None) and stock_row['grossMargins'] * 100)}")
                st.write(f"Operating: {safe_pct(stock_row.get('operatingMargins', None) and stock_row['operatingMargins'] * 100)}")
                st.write(f"Profit: {safe_pct(stock_row.get('profitMargins', None) and stock_row['profitMargins'] * 100)}")
            with fc2:
                st.markdown("**Valuation**")
                st.write(f"P/E (Trailing): {safe_val(stock_row.get('trailingPE'), '.1f')}")
                st.write(f"P/E (Forward): {safe_val(stock_row.get('forwardPE'), '.1f')}")
                st.write(f"PEG: {safe_val(stock_row.get('pegRatio'), '.2f')}")
                st.write(f"P/S: {safe_val(stock_row.get('priceToSalesTrailing12Months'), '.1f')}")
                st.write(f"EV/Rev: {safe_val(stock_row.get('enterpriseToRevenue'), '.1f')}")
            with fc3:
                st.markdown("**Growth & Earnings**")
                rg = stock_row.get("revenueGrowth")
                st.write(f"Rev Growth: {safe_pct(rg * 100 if pd.notna(rg) else None)}")
                st.write(f"EPS (Trailing): {safe_val(stock_row.get('trailingEps'), '.2f', prefix='$')}")
                st.write(f"EPS (Forward): {safe_val(stock_row.get('forwardEps'), '.2f', prefix='$')}")
            with fc4:
                st.markdown("**Balance Sheet**")
                st.write(f"Debt/Equity: {safe_val(stock_row.get('debtToEquity'), '.1f')}")
                st.write(f"ROE: {safe_pct(stock_row.get('returnOnEquity', None) and stock_row['returnOnEquity'] * 100)}")
                st.write(f"FCF: {format_large_number(stock_row.get('freeCashflow'))}")
                st.write(f"Beta: {safe_val(stock_row.get('beta'), '.2f')}")


# ─────────────────────────────────────────────
# TAB 4 — MULTI-STOCK COMPARISON (enhanced)
# ─────────────────────────────────────────────
with tab4:
    st.header("Multi-Stock Comparison")
    if len(filtered_df) == 0:
        st.warning("No data available. Adjust your filters.")
    else:
        compare_tickers = st.multiselect(
            "Select 2-10 companies to compare", sorted(filtered_df["ticker"].tolist()),
            default=sorted(filtered_df["ticker"].tolist())[:3], max_selections=10, key="compare_tickers",
        )

        if len(compare_tickers) >= 2:
            # ── Radar Chart ──
            st.subheader("Radar Profile Comparison")
            st.caption("Each axis shows the percentile rank (0-100) among all tracked companies.")
            radar_metrics = {
                "Rev Growth": "revenueGrowth", "Profit Margin": "profitMargins",
                "Rule of 40": "rule_of_40", "YTD Return": "ytd_return",
                "Valuation (low P/S)": "priceToSalesTrailing12Months",
                "FCF Yield": "fcf_yield",
            }
            # Compute percentiles from full dataset for context
            radar_pct = df.copy()
            for label, col in radar_metrics.items():
                if col in radar_pct.columns:
                    if label == "Valuation (low P/S)":
                        radar_pct[f"pct_{col}"] = 100 - pct_rank(radar_pct[col])
                    else:
                        radar_pct[f"pct_{col}"] = pct_rank(radar_pct[col])

            fig_radar = go.Figure()
            theta_labels = list(radar_metrics.keys()) + [list(radar_metrics.keys())[0]]
            for ticker in compare_tickers:
                row = radar_pct[radar_pct["ticker"] == ticker]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                r_vals = [row.get(f"pct_{col}", 50) for col in radar_metrics.values()]
                r_vals = [v if pd.notna(v) else 0 for v in r_vals]
                r_vals.append(r_vals[0])  # close the polygon
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_vals, theta=theta_labels, fill="toself", name=ticker, opacity=0.6,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                height=500, margin=dict(t=40, l=60, r=60, b=40),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # ── Normalized Price Chart ──
            st.subheader("Normalized Price Comparison (rebased to 100)")
            compare_period = PERIOD_OPTIONS[st.selectbox("Date Range", list(PERIOD_OPTIONS.keys()), index=4, key="compare_period")]
            with st.spinner("Loading comparison data..."):
                multi_data = fetch_multi_price_history(tuple(sorted(compare_tickers)), compare_period)

            if multi_data is not None and len(multi_data) > 0:
                fig_compare = go.Figure()
                for ticker in compare_tickers:
                    try:
                        if isinstance(multi_data.columns, pd.MultiIndex):
                            if ticker in multi_data.columns.get_level_values(0):
                                close = multi_data[(ticker, "Close")].dropna()
                            else:
                                continue
                        else:
                            close = multi_data["Close"].dropna()
                        if len(close) > 0:
                            normalized = (close / close.iloc[0]) * 100
                            fig_compare.add_trace(go.Scatter(
                                x=normalized.index, y=normalized.values, mode="lines", name=ticker,
                            ))
                    except Exception:
                        continue
                fig_compare.update_layout(
                    height=500, yaxis_title="Normalized Price (100 = start)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    margin=dict(t=40, l=10, r=10, b=10),
                )
                st.plotly_chart(fig_compare, use_container_width=True)

            # ── Side-by-side Metrics ──
            st.subheader("Metrics Comparison")
            compare_rows = filtered_df[filtered_df["ticker"].isin(compare_tickers)].copy()
            metrics_display = compare_rows[["ticker", "shortName", "currentPrice", "marketCap",
                                            "trailingPE", "ytd_return", "revenueGrowth",
                                            "profitMargins", "rule_of_40"]].copy()
            metrics_display.columns = ["Ticker", "Name", "Price", "Market Cap", "P/E",
                                       "YTD %", "Rev Growth", "Profit Margin", "Rule of 40"]
            st.dataframe(
                metrics_display.style.format({
                    "Price": lambda x: f"${x:.2f}" if pd.notna(x) else "N/A",
                    "Market Cap": lambda x: format_large_number(x),
                    "P/E": lambda x: safe_val(x, ".1f"), "YTD %": lambda x: safe_pct(x),
                    "Rev Growth": lambda x: safe_pct(x * 100 if pd.notna(x) else None),
                    "Profit Margin": lambda x: safe_pct(x * 100 if pd.notna(x) else None),
                    "Rule of 40": lambda x: safe_val(x, ".1f"),
                }), use_container_width=True, hide_index=True,
            )

            # ── Correlation Heatmap ──
            st.subheader("Price Correlation Heatmap")
            if multi_data is not None and len(multi_data) > 0:
                returns_dict = {}
                for ticker in compare_tickers:
                    try:
                        if isinstance(multi_data.columns, pd.MultiIndex):
                            if ticker in multi_data.columns.get_level_values(0):
                                close = multi_data[(ticker, "Close")].dropna()
                            else:
                                continue
                        else:
                            close = multi_data["Close"].dropna()
                        if len(close) > 1:
                            returns_dict[ticker] = close.pct_change().dropna()
                    except Exception:
                        continue
                if len(returns_dict) >= 2:
                    returns_df = pd.DataFrame(returns_dict).dropna()
                    corr = returns_df.corr()
                    fig_corr = px.imshow(
                        corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1,
                        title="Daily Return Correlation",
                    )
                    fig_corr.update_layout(height=500, margin=dict(t=40, l=10, r=10, b=10))
                    st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Select at least 2 companies to compare.")


# ─────────────────────────────────────────────
# TAB 5 — SAAS SCREENER
# ─────────────────────────────────────────────
with tab5:
    st.header("SaaS Screener")
    if len(filtered_df) == 0:
        st.warning("No data available. Adjust your filters.")
    else:
        sf1, sf2, sf3, sf4, sf5 = st.columns(5)
        with sf1:
            pe_range = st.slider("P/E Range", 0.0, 500.0, (0.0, 500.0), step=5.0, key="pe_range")
        with sf2:
            rev_growth_range = st.slider("Rev Growth %", -50.0, 200.0, (-50.0, 200.0), step=5.0, key="rg_range")
        with sf3:
            margin_range = st.slider("Profit Margin %", -100.0, 60.0, (-100.0, 60.0), step=5.0, key="margin_range")
        with sf4:
            rule40_min = st.slider("Min Rule of 40", -50.0, 100.0, -50.0, step=5.0, key="rule40_min")
        with sf5:
            ev_rev_range = st.slider("EV/Revenue", 0.0, 100.0, (0.0, 100.0), step=1.0, key="ev_rev_range")

        screened = filtered_df.copy()
        screened = screened[
            (screened["trailingPE"].isna())
            | ((screened["trailingPE"] >= pe_range[0]) & (screened["trailingPE"] <= pe_range[1]))
        ]
        screened = screened[
            (screened["revenueGrowth"].isna())
            | ((screened["revenueGrowth"] * 100 >= rev_growth_range[0])
               & (screened["revenueGrowth"] * 100 <= rev_growth_range[1]))
        ]
        screened = screened[
            (screened["profitMargins"].isna())
            | ((screened["profitMargins"] * 100 >= margin_range[0])
               & (screened["profitMargins"] * 100 <= margin_range[1]))
        ]
        screened = screened[
            (screened["rule_of_40"].isna()) | (screened["rule_of_40"] >= rule40_min)
        ]
        screened = screened[
            (screened["enterpriseToRevenue"].isna())
            | ((screened["enterpriseToRevenue"] >= ev_rev_range[0])
               & (screened["enterpriseToRevenue"] <= ev_rev_range[1]))
        ]

        st.markdown(f"**{len(screened)} companies match your criteria**")

        scatter_df = screened[screened["revenueGrowth"].notna() & screened["profitMargins"].notna()].copy()
        if len(scatter_df) > 0:
            scatter_df["rev_growth_pct"] = scatter_df["revenueGrowth"] * 100
            scatter_df["profit_margin_pct"] = scatter_df["profitMargins"] * 100
            scatter_df["mcap_size"] = scatter_df["marketCap"].clip(upper=scatter_df["marketCap"].quantile(0.95)).fillna(1e9)
            fig_scatter = px.scatter(
                scatter_df, x="rev_growth_pct", y="profit_margin_pct",
                size="mcap_size", color="sub_sector",
                hover_data={"ticker": True, "shortName": True, "rev_growth_pct": ":.1f",
                            "profit_margin_pct": ":.1f", "mcap_size": False},
                labels={"rev_growth_pct": "Revenue Growth %", "profit_margin_pct": "Profit Margin %"},
                title="Revenue Growth vs Profit Margin (size = Market Cap)",
            )
            fig_scatter.add_shape(
                type="line", x0=-10, y0=50, x1=50, y1=-10,
                line=dict(color="gray", width=2, dash="dash"),
            )
            fig_scatter.add_annotation(
                x=5, y=38, text="Rule of 40", showarrow=False, font=dict(color="gray", size=12),
            )
            fig_scatter.update_layout(height=600, margin=dict(t=40, l=10, r=10, b=10))
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Screened Results")
        screen_display = screened[["ticker", "shortName", "sub_sector", "currentPrice",
                                    "marketCap", "trailingPE", "ytd_return", "revenueGrowth",
                                    "profitMargins", "rule_of_40", "enterpriseToRevenue"]].copy()
        screen_display.columns = ["Ticker", "Name", "Sub-Sector", "Price", "Market Cap",
                                  "P/E", "YTD %", "Rev Growth", "Profit Margin", "Rule of 40", "EV/Revenue"]
        screen_display = screen_display.sort_values("Market Cap", ascending=False).reset_index(drop=True)
        screen_display.index = range(1, len(screen_display) + 1)
        st.dataframe(
            screen_display.style.format({
                "Price": lambda x: f"${x:.2f}" if pd.notna(x) else "N/A",
                "Market Cap": lambda x: format_large_number(x),
                "P/E": lambda x: safe_val(x, ".1f"), "YTD %": lambda x: safe_pct(x),
                "Rev Growth": lambda x: safe_pct(x * 100 if pd.notna(x) else None),
                "Profit Margin": lambda x: safe_pct(x * 100 if pd.notna(x) else None),
                "Rule of 40": lambda x: safe_val(x, ".1f"), "EV/Revenue": lambda x: safe_val(x, ".1f"),
            }),
            use_container_width=True, height=min(600, 35 * len(screen_display) + 38),
        )

        csv = screened.to_csv(index=False)
        st.download_button(
            label="Export filtered results as CSV", data=csv,
            file_name=f"saas_screener_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv",
        )


# ─────────────────────────────────────────────
# TAB 6 — MOMENTUM & SIGNALS
# ─────────────────────────────────────────────
with tab6:
    st.header("Momentum & Signals")
    if len(filtered_df) == 0:
        st.warning("No data available. Adjust your filters.")
    else:
        has_tech = all(c in filtered_df.columns for c in ["return_1m", "return_3m", "return_6m"])

        if not has_tech:
            st.warning("Technical data not available. Refresh to load.")
        else:
            # ── Relative Strength Rankings ──
            st.subheader("Relative Strength Rankings")
            st.caption("Stocks ranked by price momentum across multiple timeframes. Score = average percentile of 1M, 3M, 6M returns.")
            mom_df = filtered_df[["ticker", "shortName", "sub_sector", "currentPrice",
                                   "return_1w", "return_1m", "return_3m", "return_6m", "ytd_return"]].copy()
            mom_df["p_1m"] = pct_rank(mom_df["return_1m"])
            mom_df["p_3m"] = pct_rank(mom_df["return_3m"])
            mom_df["p_6m"] = pct_rank(mom_df["return_6m"])
            mom_df["momentum_score"] = mom_df[["p_1m", "p_3m", "p_6m"]].mean(axis=1)
            mom_df = mom_df.sort_values("momentum_score", ascending=False).reset_index(drop=True)
            mom_df.index = range(1, len(mom_df) + 1)

            mom_display = mom_df[["ticker", "shortName", "sub_sector", "momentum_score",
                                   "return_1w", "return_1m", "return_3m", "return_6m", "ytd_return"]].copy()
            mom_display.columns = ["Ticker", "Name", "Sub-Sector", "Momentum Score",
                                   "1W %", "1M %", "3M %", "6M %", "YTD %"]
            st.dataframe(
                mom_display.style.format({
                    "Momentum Score": "{:.1f}", "1W %": lambda x: safe_pct(x),
                    "1M %": lambda x: safe_pct(x), "3M %": lambda x: safe_pct(x),
                    "6M %": lambda x: safe_pct(x), "YTD %": lambda x: safe_pct(x),
                }).background_gradient(subset=["Momentum Score"], cmap="RdYlGn", vmin=0, vmax=100),
                use_container_width=True, height=min(600, 35 * len(mom_display) + 38),
            )

            # ── Moving Average Dashboard ──
            st.subheader("Moving Average Dashboard")
            ma_cols = [c for c in ["ma_50", "ma_200", "above_ma50", "above_ma200", "signal"] if c in filtered_df.columns]
            if len(ma_cols) >= 2:
                ma_df = filtered_df[["ticker", "shortName", "sub_sector", "currentPrice"] + ma_cols].copy()
                ma_df = ma_df[ma_df["ma_50"].notna()].copy()

                # Summary metrics
                m1, m2, m3, m4 = st.columns(4)
                if "above_ma50" in ma_df.columns:
                    above_50 = ma_df["above_ma50"].sum()
                    m1.metric("Above 50-Day MA", f"{above_50} / {len(ma_df)}")
                if "above_ma200" in ma_df.columns:
                    above_200_df = ma_df[ma_df["above_ma200"].notna()]
                    above_200 = above_200_df["above_ma200"].sum()
                    m2.metric("Above 200-Day MA", f"{int(above_200)} / {len(above_200_df)}")
                if "signal" in ma_df.columns:
                    golden = len(ma_df[ma_df["signal"] == "Golden Cross"])
                    death = len(ma_df[ma_df["signal"] == "Death Cross"])
                    m3.metric("Recent Golden Crosses", str(golden))
                    m4.metric("Recent Death Crosses", str(death))

                # MA table
                ma_display = ma_df.copy()
                rename_map = {"ticker": "Ticker", "shortName": "Name", "sub_sector": "Sub-Sector",
                              "currentPrice": "Price"}
                if "ma_50" in ma_display.columns:
                    rename_map["ma_50"] = "50-Day MA"
                if "ma_200" in ma_display.columns:
                    rename_map["ma_200"] = "200-Day MA"
                if "above_ma50" in ma_display.columns:
                    rename_map["above_ma50"] = "Above 50D"
                if "above_ma200" in ma_display.columns:
                    rename_map["above_ma200"] = "Above 200D"
                if "signal" in ma_display.columns:
                    rename_map["signal"] = "Signal"
                ma_display = ma_display.rename(columns=rename_map)
                format_dict = {"Price": "${:.2f}"}
                if "50-Day MA" in ma_display.columns:
                    format_dict["50-Day MA"] = "${:.2f}"
                if "200-Day MA" in ma_display.columns:
                    format_dict["200-Day MA"] = "${:.2f}"
                st.dataframe(
                    ma_display.style.format(format_dict),
                    use_container_width=True, height=min(600, 35 * len(ma_display) + 38),
                )

                # ── Breadth chart: % above MA over sectors ──
                st.subheader("Market Breadth by Sector")
                if "above_ma50" in filtered_df.columns:
                    breadth = filtered_df[filtered_df["above_ma50"].notna()].groupby("sub_sector").agg(
                        total=("above_ma50", "count"),
                        above_50=("above_ma50", "sum"),
                    ).reset_index()
                    if "above_ma200" in filtered_df.columns:
                        breadth_200 = filtered_df[filtered_df["above_ma200"].notna()].groupby("sub_sector").agg(
                            above_200=("above_ma200", "sum"),
                        ).reset_index()
                        breadth = breadth.merge(breadth_200, on="sub_sector", how="left")
                    breadth["above_50"] = pd.to_numeric(breadth["above_50"], errors="coerce")
                    breadth["total"] = pd.to_numeric(breadth["total"], errors="coerce")
                    breadth["pct_above_50"] = (breadth["above_50"] / breadth["total"] * 100).round(1)
                    if "above_200" in breadth.columns:
                        breadth["above_200"] = pd.to_numeric(breadth["above_200"], errors="coerce")
                        breadth["pct_above_200"] = (breadth["above_200"] / breadth["total"] * 100).round(1)

                    fig_breadth = go.Figure()
                    fig_breadth.add_trace(go.Bar(
                        x=breadth["sub_sector"], y=breadth["pct_above_50"],
                        name="% Above 50-Day MA", marker_color="#2962FF",
                    ))
                    if "pct_above_200" in breadth.columns:
                        fig_breadth.add_trace(go.Bar(
                            x=breadth["sub_sector"], y=breadth["pct_above_200"],
                            name="% Above 200-Day MA", marker_color="#f59e0b",
                        ))
                    fig_breadth.update_layout(
                        barmode="group", height=400, yaxis_title="% of Stocks",
                        margin=dict(t=20, l=10, r=10, b=10),
                    )
                    st.plotly_chart(fig_breadth, use_container_width=True)
            else:
                st.info("Moving average data not available.")

            # ── Max Drawdown Rankings ──
            if "max_drawdown_6m" in filtered_df.columns:
                st.subheader("6-Month Max Drawdown")
                st.caption("Largest peak-to-trough decline in the past 6 months. Less negative = more resilient.")
                dd_df = filtered_df[filtered_df["max_drawdown_6m"].notna()].sort_values("max_drawdown_6m").copy()
                if len(dd_df) > 0:
                    dd_view = st.radio("Show", ["Worst 25", "Best 25", "All"], horizontal=True, key="dd_view")
                    if dd_view == "Worst 25":
                        dd_plot = dd_df.head(25)
                    elif dd_view == "Best 25":
                        dd_plot = dd_df.tail(25)
                    else:
                        dd_plot = dd_df
                    fig_dd = go.Figure()
                    colors = ["#ef4444" if v < -20 else "#f59e0b" if v < -10 else "#22c55e" for v in dd_plot["max_drawdown_6m"]]
                    fig_dd.add_trace(go.Bar(
                        y=dd_plot["ticker"], x=dd_plot["max_drawdown_6m"], orientation="h",
                        marker_color=colors,
                        text=dd_plot["max_drawdown_6m"].apply(lambda x: f"{x:.1f}%"),
                        textposition="outside",
                    ))
                    fig_dd.update_layout(
                        height=max(400, 22 * len(dd_plot)), xaxis_title="Max Drawdown %",
                        margin=dict(t=20, l=10, r=10, b=10),
                    )
                    st.plotly_chart(fig_dd, use_container_width=True)
