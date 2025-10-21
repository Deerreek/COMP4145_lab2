import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.optimize
import streamlit as st
from io import BytesIO

# ================= EASY =================

def portfolio_summary(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary table: Last Price, Mean Daily Return (%), Volatility (%), Max Drawdown (%).
    """
    summary = pd.DataFrame(index=prices.columns)
    summary['Last Price'] = prices.iloc[-1]
    summary['Mean Daily Return (%)'] = returns.mean() * 100
    summary['Volatility (%)'] = returns.std() * 100
    # Max drawdown
    cum_returns = (1 + returns).cumprod()
    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max
    summary['Max Drawdown (%)'] = drawdown.min() * 100
    summary.index.name = 'Ticker'
    return summary

@st.cache_data
def get_portfolio_summary(prices, returns):
    return portfolio_summary(prices, returns)


def top_bottom_movers(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute total return for each ticker, return sorted DataFrame.
    """
    total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
    movers = pd.DataFrame({'Total Return (%)': total_return})
    movers = movers.sort_values('Total Return (%)', ascending=False)
    return movers

@st.cache_data
def get_top_bottom_movers(prices):
    return top_bottom_movers(prices)

# ================= MODERATE =================

def correlation_heatmap(returns: pd.DataFrame, abs_corr: bool = False):
    """
    Compute and plot correlation heatmap of daily returns.
    """
    corr = returns.corr()
    if abs_corr:
        corr = corr.abs()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu',
        aspect='auto',
        title='Correlation Heatmap',
    )
    fig.update_layout(height=500)
    return fig


def allocation_pie(selected_tickers, weights):
    """
    Pie chart of allocation weights.
    """
    fig = go.Figure(
        data=[go.Pie(labels=selected_tickers, values=weights, hole=0.4)]
    )
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(title='Portfolio Allocation', height=400)
    return fig


def weighted_portfolio_curve(returns: pd.DataFrame, weights: np.ndarray):
    """
    Compute weighted portfolio cumulative return curve.
    """
    weighted_returns = returns @ weights
    cum_curve = (1 + weighted_returns).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cum_curve.index, y=cum_curve, mode='lines', name='Portfolio'))
    fig.update_layout(title='Weighted Portfolio Cumulative Return', yaxis_title='Growth', height=400)
    return fig

# ================= COMPLEX =================

def risk_return_scatter(returns: pd.DataFrame):
    """
    Scatter plot: x=volatility, y=mean return, size/color by Sharpe ratio.
    """
    mean = returns.mean() * 100
    vol = returns.std() * 100
    sharpe = mean / vol.replace(0, np.nan)
    # Clamp marker size to positive values for Plotly
    marker_size = sharpe.copy()
    marker_size[marker_size <= 0] = 0.1
    df = pd.DataFrame({'Mean': mean, 'Volatility': vol, 'Sharpe': sharpe, 'MarkerSize': marker_size})
    df['Ticker'] = df.index
    fig = px.scatter(
        df,
        x='Volatility',
        y='Mean',
        size='MarkerSize',
        color='Sharpe',
        hover_name='Ticker',
        title='Risk–Return Scatter',
        height=500,
    )
    # 45º reference line
    fig.add_shape(type='line', x0=0, y0=0, x1=df['Volatility'].max(), y1=df['Volatility'].max(),
                  line=dict(dash='dash', color='gray'))
    return fig


def markowitz_optimization(returns: pd.DataFrame):
    """
    Maximize Sharpe ratio (annualized) with constraints: sum(weights)=1, weights>=0.
    Returns optimal weights, expected return, volatility, Sharpe.
    """
    mean = returns.mean() * 252
    cov = returns.cov() * 252
    n = len(mean)
    def neg_sharpe(weights):
        port_return = np.dot(weights, mean)
        port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
        return -port_return / port_vol if port_vol > 0 else 1e6
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1)] * n
    x0 = np.ones(n) / n
    result = scipy.optimize.minimize(neg_sharpe, x0, bounds=bounds, constraints=cons)
    weights = result.x
    port_return = np.dot(weights, mean)
    port_vol = np.sqrt(np.dot(weights, np.dot(cov, weights)))
    sharpe = port_return / port_vol if port_vol > 0 else np.nan
    return weights, port_return, port_vol, sharpe


def optimization_tab(returns: pd.DataFrame, tickers: list, set_weights_callback):
    """
    Markowitz optimization tab UI and logic.
    """
    st.subheader('Markowitz Portfolio Optimization')
    weights, port_return, port_vol, sharpe = markowitz_optimization(returns)
    df = pd.DataFrame({'Ticker': tickers, 'Optimal Weight': weights})
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.metric('Expected Annual Return', f'{port_return*100:.2f}%')
    st.metric('Expected Volatility', f'{port_vol*100:.2f}%')
    st.metric('Sharpe Ratio', f'{sharpe:.2f}')
    if st.button('Apply optimal weights'):
        set_weights_callback(weights)
        st.success('Optimal weights applied to Allocation tab!')
