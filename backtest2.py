# Code by Grok (grok interface)

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def calculate_rsi(data, periods=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def backtest_rsi_strategy(ticker='BTC-USD', start_date='2017-01-01', end_date='2025-02-20', 
                         rsi_period=14, buy_threshold=25, sell_threshold=75):
    # Fetch weekly historical data
    df = yf.download(ticker, start=start_date, end=end_date, interval='1wk')
    
    # Calculate RSI
    df['RSI'] = calculate_rsi(df['Close'], rsi_period)
    
    # Initialize columns
    df['Position'] = 0  # 1 for long, 0 for no position
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = 0.0
    
    # Backtest logic
    in_position = False
    
    for i in range(1, len(df)):
        current_rsi = df['RSI'].iloc[i].item()  # Get scalar value
        prev_rsi = df['RSI'].iloc[i-1].item()  # Get scalar value
        
        # Buy signal
        if not in_position and current_rsi < buy_threshold:
            df.loc[df.index[i], 'Position'] = 1
            in_position = True
            price = df['Close'].iloc[i].item()  # Get scalar value
            print(f"Buy at {df.index[i]} - Price: {price:.2f}, RSI: {current_rsi:.2f}")
        
        # Sell signal
        elif in_position and current_rsi > sell_threshold:
            df.loc[df.index[i], 'Position'] = 0
            in_position = False
            price = df['Close'].iloc[i].item()  # Get scalar value
            print(f"Sell at {df.index[i]} - Price: {price:.2f}, RSI: {current_rsi:.2f}")
        
        # Maintain position
        if in_position:
            df.loc[df.index[i], 'Position'] = 1
    
    # Calculate returns
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Returns']
    
    # Performance metrics
    total_return = (df['Strategy_Returns'] + 1).prod() - 1
    buy_and_hold_return = (df['Close'].iloc[-1].item() / df['Close'].iloc[0].item()) - 1
    trades = len(df[df['Position'].diff().abs() == 1]) // 2
    win_rate = len(df[df['Strategy_Returns'] > 0]) / len(df[df['Strategy_Returns'] != 0]) if trades > 0 else 0
    
    # Additional metrics
    variance = df['Strategy_Returns'].var() * 52  # Annualized variance
    mean_return = df['Strategy_Returns'].mean() * 52  # Annualized mean return
    std_dev = df['Strategy_Returns'].std() * np.sqrt(52)  # Annualized std dev
    sharpe_ratio = mean_return / std_dev if std_dev != 0 else 0
    max_drawdown = calculate_max_drawdown(df['Strategy_Returns'].dropna())
    years = (df.index[-1] - df.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # Summary
    print("\nPerformance Summary:")
    print(f"Period: {start_date} to {end_date}")
    print(f"Total Strategy Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Buy & Hold Return: {buy_and_hold_return:.2%}")
    print(f"Number of Trades: {trades}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Annualized Variance: {variance:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Starting Price: ${df['Close'].iloc[0].item():.2f}")
    print(f"Ending Price: ${df['Close'].iloc[-1].item():.2f}")
    
    # Key lessons
    print("\nKey Lessons for Actual Trading:")
    print("1. RSI thresholds (25/75) may need adjustment based on market conditions")
    print("2. Transaction costs not included - would reduce returns")
    print("3. Weekly timeframe reduces noise but may miss shorter-term opportunities")
    print("4. Past performance doesn't guarantee future results")
    print("5. Risk management (stop-losses, position sizing) is critical")
    print("6. High variance indicates significant volatility in returns")
    print("7. Sharpe Ratio helps evaluate risk-adjusted returns")
    
    return df

# Execute backtest
if __name__ == "__main__":
    # Install required package if needed: pip install yfinance
    result_df = backtest_rsi_strategy()