import pandas as pd
import streamlit as st
import yfinance as yf

DEFAULT_TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "JPM",
    "NVDA",
    "XOM",
    "PFE",
]


@st.cache_data(show_spinner=False)
def get_prices(tickers, period="1y"):
    """Download adjusted close prices for the given tickers."""
    if not tickers:
        raise ValueError("Select at least one ticker.")

    tickers_tuple = tuple(sorted(set(tickers)))
    try:
        raw = yf.download(
            tickers=list(tickers_tuple),
            period=period.lower(),
            group_by="ticker",
            auto_adjust=False,
            progress=False,
        )
    except Exception as exc:
        raise ValueError(f"Failed to download data: {exc}") from exc

    if raw.empty:
        raise ValueError("No price data returned. Please try a shorter period.")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))
        if "Adj Close" in level0:
            prices = raw.loc[:, "Adj Close"]
        elif "Adj Close" in level1:
            prices = raw.xs("Adj Close", axis=1, level=1)
        elif "Close" in level0:
            prices = raw.loc[:, "Close"]
        elif "Close" in level1:
            prices = raw.xs("Close", axis=1, level=1)
        else:
            raise ValueError("Adjusted close prices not available for the selected tickers.")
    else:
        adj_close = raw.get("Adj Close") or raw.get("Close")
        if adj_close is None:
            raise ValueError("Adjusted close prices not available for the selected tickers.")
        if isinstance(adj_close, pd.Series):
            prices = adj_close.to_frame(name=tickers_tuple[0])
        else:
            prices = adj_close

    prices = prices.sort_index().ffill().dropna(how="all")
    return prices


def get_returns(prices):
    """Calculate daily percentage returns."""
    if prices.empty:
        return pd.DataFrame(columns=prices.columns)
    returns = prices.pct_change().dropna(how="all")
    return returns


def get_cumreturns(prices):
    """Calculate cumulative returns from the first available price."""
    if prices.empty:
        return pd.DataFrame(columns=prices.columns)
    baseline = prices.iloc[0]
    cum_returns = (prices / baseline) - 1
    return cum_returns.dropna(how="all")


def summary_table(returns, prices):
    """Build a summary table with last price, mean return, and volatility."""
    if prices.empty:
        return pd.DataFrame(columns=["Ticker", "Last Price", "Mean Daily Return (%)", "Volatility (%)"])

    last_prices = prices.ffill().iloc[-1]
    mean_daily = returns.mean() * 100 if not returns.empty else pd.Series(0, index=last_prices.index)
    volatility = returns.std() * 100 if not returns.empty else pd.Series(0, index=last_prices.index)

    summary = pd.DataFrame({
        "Ticker": last_prices.index,
        "Last Price": last_prices.values,
        "Mean Daily Return (%)": mean_daily.reindex(last_prices.index).values,
        "Volatility (%)": volatility.reindex(last_prices.index).values,
    })
    return summary
