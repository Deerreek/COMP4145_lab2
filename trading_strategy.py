import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json

# Download 5 years of MSFT data
def get_stock_data(ticker, period="5y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

# Calculate moving averages
def calculate_moving_averages(data):
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    return data

# Identify golden cross (buy signals)
def identify_golden_cross(data):
    data['Signal'] = 0  # Initialize signal column with 0
    # Golden Cross occurs when MA50 crosses above MA200
    data['GoldenCross'] = (data['MA50'] > data['MA200']) & (data['MA50'].shift(1) <= data['MA200'].shift(1))
    return data

# Implement trading strategy
def implement_strategy(data):
    positions = []

    # Need at least 200 days to calculate the 200-day MA
    data = data.iloc[200:].copy()

    buy_dates = data[data['GoldenCross'] == True].index.tolist()

    for buy_date in buy_dates:
        # Get buy price
        buy_price = data.loc[buy_date, 'Close']

        # Calculate target sell price (15% profit)
        target_price = buy_price * 1.15

        # Set maximum holding period
        max_sell_date = buy_date + pd.Timedelta(days=60)

        # Get data slice for potential sell period
        sell_period = data.loc[buy_date:max_sell_date].copy()

        # Check if target price is reached during the period
        target_reached = sell_period[sell_period['Close'] >= target_price]

        if not target_reached.empty:
            # Sell at first date target is reached
            sell_date = target_reached.index[0]
            sell_price = target_reached.loc[sell_date, 'Close']
            sell_reason = "Target reached"
        else:
            # Sell at end of maximum holding period
            sell_date_candidates = sell_period.index.tolist()
            if sell_date_candidates:
                sell_date = sell_date_candidates[-1]
                sell_price = data.loc[sell_date, 'Close']
                sell_reason = "Max holding period"
            else:
                # Skip if no valid sell date (should not happen in practice)
                continue

        # Calculate holding period in calendar days
        holding_days = (sell_date - buy_date).days

        # Calculate profit
        profit_pct = (sell_price / buy_price - 1) * 100

        positions.append({
            'BuyDate': buy_date,
            'BuyPrice': buy_price,
            'SellDate': sell_date,
            'SellPrice': sell_price,
            'HoldingDays': holding_days,
            'ProfitPct': profit_pct,
            'SellReason': sell_reason
        })

    return pd.DataFrame(positions)

# Analyze the results
def analyze_results(positions):
    if positions.empty:
        return "No trading signals detected"

    # Summary statistics
    total_trades = len(positions)
    win_trades = len(positions[positions['ProfitPct'] > 0])
    loss_trades = total_trades - win_trades
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

    avg_profit = positions['ProfitPct'].mean()
    avg_win = positions[positions['ProfitPct'] > 0]['ProfitPct'].mean() if win_trades > 0 else 0
    avg_loss = positions[positions['ProfitPct'] <= 0]['ProfitPct'].mean() if loss_trades > 0 else 0

    avg_holding = positions['HoldingDays'].mean()

    target_reached = len(positions[positions['SellReason'] == 'Target reached'])
    max_period = len(positions[positions['SellReason'] == 'Max holding period'])

    print("\n===== Trading Strategy Results (Golden Cross)=====")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {win_trades} ({win_rate:.2f}%)")
    print(f"Losing Trades: {loss_trades}")
    print(f"Average Profit: {avg_profit:.2f}%")

    return positions

# Main function
def main():
    # Get stock data
    data = get_stock_data("MSFT")

    # Calculate moving averages
    data = calculate_moving_averages(data)

    # Identify golden cross
    data = identify_golden_cross(data)

    # Implement strategy
    positions = implement_strategy(data)

    # Analyze results
    analyze_results(positions)

    # Export data for frontend
    export_data_for_frontend(data, positions)

    # Return the detailed positions dataframe
    return positions

def export_data_for_frontend(data, positions):
    """Export trading data to JSON files for frontend consumption"""
    
    # Prepare price data with moving averages
    price_data = data[['Close', 'MA50', 'MA200']].copy()
    price_data = price_data.dropna()
    price_data.index = price_data.index.strftime('%Y-%m-%d')
    
    price_json = {
        'dates': price_data.index.tolist(),
        'close': price_data['Close'].tolist(),
        'ma50': price_data['MA50'].tolist(),
        'ma200': price_data['MA200'].tolist()
    }
    
    # Prepare buy/sell points
    buy_sell_points = {
        'buy_dates': positions['BuyDate'].dt.strftime('%Y-%m-%d').tolist(),
        'buy_prices': positions['BuyPrice'].tolist(),
        'sell_dates': positions['SellDate'].dt.strftime('%Y-%m-%d').tolist(),
        'sell_prices': positions['SellPrice'].tolist()
    }
    
    # Prepare trade statistics
    total_trades = len(positions)
    win_trades = len(positions[positions['ProfitPct'] > 0])
    loss_trades = total_trades - win_trades
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0
    
    avg_profit = positions['ProfitPct'].mean()
    avg_win = positions[positions['ProfitPct'] > 0]['ProfitPct'].mean() if win_trades > 0 else 0
    avg_loss = positions[positions['ProfitPct'] <= 0]['ProfitPct'].mean() if loss_trades > 0 else 0
    avg_holding = positions['HoldingDays'].mean()
    
    total_return = positions['ProfitPct'].sum()
    max_profit = positions['ProfitPct'].max()
    max_loss = positions['ProfitPct'].min()
    
    statistics = {
        'total_trades': int(total_trades),
        'winning_trades': int(win_trades),
        'losing_trades': int(loss_trades),
        'win_rate': float(win_rate),
        'avg_profit': float(avg_profit),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'avg_holding_days': float(avg_holding),
        'total_return': float(total_return),
        'max_profit': float(max_profit),
        'max_loss': float(max_loss),
        'target_reached': int(len(positions[positions['SellReason'] == 'Target reached'])),
        'max_period': int(len(positions[positions['SellReason'] == 'Max holding period']))
    }
    
    # Prepare detailed trades
    trades_list = []
    for idx, row in positions.iterrows():
        trades_list.append({
            'buy_date': row['BuyDate'].strftime('%Y-%m-%d'),
            'buy_price': float(row['BuyPrice']),
            'sell_date': row['SellDate'].strftime('%Y-%m-%d'),
            'sell_price': float(row['SellPrice']),
            'holding_days': int(row['HoldingDays']),
            'profit_pct': float(row['ProfitPct']),
            'sell_reason': row['SellReason']
        })
    
    # Save to JSON files
    with open('price_data.json', 'w') as f:
        json.dump(price_json, f)
    
    with open('buy_sell_points.json', 'w') as f:
        json.dump(buy_sell_points, f)
    
    with open('statistics.json', 'w') as f:
        json.dump(statistics, f)
    
    with open('trades.json', 'w') as f:
        json.dump(trades_list, f)
    
    print("\n✓ Data exported to JSON files for frontend")

positions = main()
print("\nDetailed Trades:")
print(positions.to_string())