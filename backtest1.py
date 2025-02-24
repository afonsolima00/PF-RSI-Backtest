# Code by Claude 3.5 Sonnet (inside copilot)

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def calculate_rsi(data, periods=14):
    # Calculate price changes
    delta = data.diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    excess_returns = returns - risk_free_rate/52  # Converting annual rate to weekly
    return np.sqrt(52) * (excess_returns.mean() / excess_returns.std())

def main():
    # Step 1: Get BTC-USD data from Yahoo Finance (weekly data since 2017)
    print("Fetching BTC-USD historical data...")
    btc = yf.download('BTC-USD', 
                      start='2017-01-01',
                      end=datetime.now().strftime('%Y-%m-%d'),
                      interval='1wk')
    
    # Step 2: Calculate RSI
    print("Calculating RSI...")
    btc['RSI'] = calculate_rsi(btc['Close'], periods=14)
    
    # Step 3: Implement trading strategy
    print("Implementing trading strategy...")
    btc['Position'] = 0  # 0 means no position, 1 means long position
    btc['Signal'] = 0    # 1 for buy signal, -1 for sell signal
    
    # Generate trading signals
    btc.loc[btc['RSI'] < 25, 'Signal'] = 1    # Buy signal
    btc.loc[btc['RSI'] > 75, 'Signal'] = -1   # Sell signal
    
    # Initialize position column
    position = 0
    positions = []
    
    # Implement position logic
    for signal in btc['Signal']:
        if signal == 1 and position == 0:  # Buy signal and no position
            position = 1
        elif signal == -1 and position == 1:  # Sell signal and long position
            position = 0
        positions.append(position)
    
    btc['Position'] = positions
    
    # Calculate returns
    btc['Returns'] = btc['Close'].pct_change()
    btc['Strategy_Returns'] = btc['Position'].shift(1) * btc['Returns']
    
    # Step 4: Calculate performance metrics
    print("\nPerformance Metrics:")
    print("-" * 50)
    
    # Total Return
    total_return = (1 + btc['Strategy_Returns'].dropna()).cumprod()[-1] - 1
    print(f"Total Return: {total_return:.2%}")
    
    # Annualized Return
    n_years = len(btc) / 52
    annualized_return = (1 + total_return) ** (1/n_years) - 1
    print(f"Annualized Return: {annualized_return:.2%}")
    
    # Maximum Drawdown
    cum_returns = (1 + btc['Strategy_Returns'].dropna()).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns/rolling_max - 1
    max_drawdown = drawdowns.min()
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    
    # Sharpe Ratio
    sharpe = calculate_sharpe_ratio(btc['Strategy_Returns'].dropna())
    print(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Trading Statistics
    n_trades = (btc['Signal'] != 0).sum()
    print(f"Number of Trades: {n_trades}")
    
    # Variance of returns
    variance = btc['Strategy_Returns'].var()
    print(f"Variance of Returns: {variance:.4f}")
    
    # Save results to CSV for further analysis
    btc.to_csv('backtest_results.csv')
    print("\nResults have been saved to 'backtest_results.csv'")

if __name__ == "__main__":
    main()