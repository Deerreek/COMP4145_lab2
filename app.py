import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from portfolio_features import (
    get_portfolio_summary,
    get_top_bottom_movers,
    correlation_heatmap,
    allocation_pie,
    weighted_portfolio_curve,
    risk_return_scatter,
    optimization_tab,
)
from utils.data_loader import (
    DEFAULT_TICKERS,
    get_cumreturns,
    get_prices,
    get_returns,
    summary_table,
)

# Page configuration
st.set_page_config(
    page_title="Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for subtle theming
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    h1 {
        color: #667eea;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
    }
    h2 {
        color: #764ba2;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_price_chart(prices):
    """Render line chart for adjusted closing prices."""
    fig = go.Figure()
    for ticker in prices.columns:
        fig.add_trace(
            go.Scatter(
                x=prices.index,
                y=prices[ticker],
                mode="lines",
                name=ticker,
                hovertemplate="Date=%{x|%Y-%m-%d}<br>Price=$%{y:,.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Adjusted Closing Prices",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        height=600,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_cumreturn_chart(cum_returns):
    """Render line chart for cumulative returns."""
    cum_percent = cum_returns * 100
    fig = go.Figure()
    for ticker in cum_percent.columns:
        fig.add_trace(
            go.Scatter(
                x=cum_percent.index,
                y=cum_percent[ticker],
                mode="lines",
                name=ticker,
                hovertemplate="Date=%{x|%Y-%m-%d}<br>Cumulative Return=%{y:.2f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode="x unified",
        height=600,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("📈 Trading Strategy Dashboard")
    st.markdown("**Multi-Stock Performance Overview**")

    # --- Sidebar ---
    st.sidebar.header("⚙️ Settings")
    selected_tickers = st.sidebar.multiselect(
        "Select Tickers",
        options=DEFAULT_TICKERS,
        default=DEFAULT_TICKERS,
        key="ticker_select",
    )
    period = st.sidebar.selectbox("Period", ["YTD", "1y", "3y", "5y"], index=1)
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        get_prices.clear()
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.info("""
        - Choose one or more tickers to compare side-by-side.
        - Cumulative returns are relative to the first selected day.
    """)
    # Allocation sliders
    st.sidebar.subheader("Portfolio Weights")
    if selected_tickers:
        default_weights = np.ones(len(selected_tickers)) / len(selected_tickers)
        weights = []
        for i, ticker in enumerate(selected_tickers):
            w = st.sidebar.slider(f"{ticker} weight", 0.0, 1.0, float(default_weights[i]), 0.01)
            weights.append(w)
        weights = np.array(weights)
        if weights.sum() != 1.0:
            weights = weights / weights.sum()
            st.sidebar.caption("Weights normalized to sum to 1.")
    else:
        weights = np.array([])

    if not selected_tickers:
        st.warning("Please select at least one ticker to view data.")
        return

    with st.spinner("Loading data..."):
        try:
            prices = get_prices(selected_tickers, period=period)
        except ValueError as err:
            st.error(err)
            return

    available = [ticker for ticker in selected_tickers if ticker in prices.columns]
    missing = [ticker for ticker in selected_tickers if ticker not in prices.columns]

    if not available:
        st.error("No price data returned for the selected tickers.")
        return

    if missing:
        st.warning(f"No data returned for: {', '.join(missing)}")

    prices = prices[available]
    returns = get_returns(prices)
    cum_returns = get_cumreturns(prices)

    # --- Tabs ---
    tab_names = [
        "Prices", "Cumulative Return", "Summary", "Top vs Bottom", "Correlation", "Allocation", "Risk–Return", "Optimization"
    ]
    (
        prices_tab, cum_tab, summary_tab, movers_tab, corr_tab, alloc_tab, risk_tab, opt_tab
    ) = st.tabs(tab_names)

    with prices_tab:
        render_price_chart(prices)

    with cum_tab:
        if cum_returns.empty:
            st.info("Not enough data to compute cumulative returns for the selected period.")
        else:
            render_cumreturn_chart(cum_returns)

    with summary_tab:
        st.subheader("Portfolio Summary Table")
        summary_df = get_portfolio_summary(prices, returns)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        csv = summary_df.to_csv().encode('utf-8')
        st.download_button("Download Summary CSV", csv, "portfolio_summary.csv", "text/csv")

    with movers_tab:
        st.subheader("Top & Bottom Movers")
        movers = get_top_bottom_movers(prices)
        st.write("Top 3 Movers:")
        st.dataframe(movers.head(3), use_container_width=True, hide_index=True)
        st.write("Bottom 3 Movers:")
        st.dataframe(movers.tail(3), use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=movers.index, y=movers['Total Return (%)'], marker_color=["green" if x >=0 else "red" for x in movers['Total Return (%)']]))
        fig.update_layout(title="Total Return by Ticker", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with corr_tab:
        st.subheader("Correlation Heatmap")
        abs_corr = st.checkbox("Show absolute correlation", value=False)
        fig = correlation_heatmap(returns, abs_corr)
        st.plotly_chart(fig, use_container_width=True)

    with alloc_tab:
        st.subheader("Portfolio Allocation")
        if len(weights) == len(selected_tickers) and len(weights) > 0:
            pie_fig = allocation_pie(selected_tickers, weights)
            st.plotly_chart(pie_fig, use_container_width=True)
            curve_fig = weighted_portfolio_curve(returns, weights)
            st.plotly_chart(curve_fig, use_container_width=True)
        else:
            st.info("Select tickers and adjust weights to view allocation.")

    with risk_tab:
        st.subheader("Risk–Return Scatter")
        fig = risk_return_scatter(returns)
        st.plotly_chart(fig, use_container_width=True)

    with opt_tab:
        def set_weights_callback(new_weights):
            st.session_state['weights'] = new_weights
        optimization_tab(returns, selected_tickers, set_weights_callback)


if __name__ == "__main__":
    main()
