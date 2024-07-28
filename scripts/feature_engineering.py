import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def reverse_dataframe(df):
    """Reverse the order of DataFrame rows."""
    return df.iloc[::-1].reset_index(drop=True)


def calculate_moving_averages(df, windows):
    """Calculate simple and exponential moving averages for specified windows."""
    for window in windows:
        df[f'SMA_{window}'] = df['close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['close'].ewm(span=window, adjust=False).mean()

def relative_strength_index(df, window=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.ewm(span=window, adjust=False).mean()
    avg_loss = loss.ewm(span=window, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

def calculate_macd(df, slow=26, fast=12, signal=9):
    """Calculate Moving Average Convergence Divergence (MACD)."""
    df['EMA_Fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['EMA_Slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_Fast'] - df['EMA_Slow']
    df['Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()

def calculate_bollinger_bands(df, window=20):
    """Calculate Bollinger Bands."""
    df['SMA_20'] = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    df['Upper_Band'] = df['SMA_20'] + (2 * rolling_std)
    df['Lower_Band'] = df['SMA_20'] - (2 * rolling_std)

def calculate_historical_volatility(df, window=30):
    """Calculate historical volatility."""
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['rolling_volatility'] = df['log_returns'].rolling(window=window).std() * np.sqrt(252)

def calculate_momentum(df, n=10):
    """Calculate momentum."""
    df['momentum'] = df['close'] - df['close'].shift(n)

def main(ticker):

    print("Engineering features ... ")
    file = f'data/{ticker}_daily_data.csv'

    df = pd.read_csv(file)
    df = reverse_dataframe(df)
    calculate_moving_averages(df, [10, 20, 50, 100, 200])
    relative_strength_index(df)
    calculate_macd(df)
    calculate_bollinger_bands(df)
    calculate_historical_volatility(df)
    calculate_momentum(df)
    df = reverse_dataframe(df)
    df.to_csv(file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch data for a specific stock ticker.')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    args = parser.parse_args()

    main(args.ticker)