from datetime import datetime, timedelta
from flask import json
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import boto3
from io import BytesIO, StringIO
import csv
import requests
import os
import torch.nn as nn
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
s3 = boto3.client('s3')
BUCKET_NAME = 'faangfinance'

class LSTMModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_features, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return out


SPLITS = {
    'AAPL': {
        '2020-08-28': 4,
        '2014-06-09': 7,
        '2005-02-28': 2,
        '2000-06-21': 2
    },
    'GOOG': {
        '2022-07-18': 20,
        '2015-04-27': 1.0027455,
        '2014-03-27': 2.002
    },
    'META': {
    },
    'NFLX': {
        '2015-07-15': 7,
        '2004-02-12': 2
    },
    'AMZN': {
        '2022-06-06': 20,
    },
    'NVDA': {
        '2024-06-10': 10,
        '2021-07-20': 4,
        '2007-09-11': 1.5,
        '2006-04-07': 2,
        '2001-09-17': 2,
        '2000-06-27': 2
    },
    'MSFT': {
        '2003-02-18': 2,
    },
    'ADBE': {
        '2005-05-24': 2,
        '2000-10-25': 2,
    }
}

def load_from_s3(bucket_name, s3_path):
    obj = s3.get_object(Bucket=bucket_name, Key=s3_path)
    return obj['Body'].read()


def load_model(ticker):
    model_path = f'models/{ticker}_model.pth'
    model_data = load_from_s3(BUCKET_NAME, model_path)
    model = LSTMModel(num_features=26, hidden_dim=25,
                      num_layers=2, output_size=1)
    model.load_state_dict(torch.load(BytesIO(model_data)))
    model.eval()
    return model


def load_scaler(scaler_type, ticker):
    scaler_path = f'data/{ticker}_scalers/{scaler_type}.pkl'
    scaler_data = load_from_s3(BUCKET_NAME, scaler_path)
    return joblib.load(BytesIO(scaler_data))


def load_ss(ticker):
    return load_scaler('ss', ticker)


def load_mm(ticker):
    return load_scaler('mm', ticker)


def load_vss(ticker):
    return load_scaler('vss', ticker)


def load_css(ticker):
    return load_scaler('css', ticker)

def preprocess_input(df, ss, mm, vss):
    logger.info("Starting input preprocessing")

    # Drop 'time_stamp' and reset index
    df = df.loc[:10]
    logger.info(f"Original DataFrame: {df.head()}")
    logger.info(f"Shape of DataFrame before tensor conversion: {df.shape}")
    df = df.drop('time_stamp', axis=1)
    logger.info(f"Shape of DataFrame after date drop: {df.shape}")
    logger.info(f"Date drop DataFrame: {df.head()}")

    df = df.reset_index(drop=True)

    # Convert 'volume' to float
    df['volume'] = df['volume'].astype(float)

    ss_features = ['open', 'high', 'low', 'close', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20',
                   'SMA_50', 'EMA_50', 'SMA_100', 'EMA_100', 'SMA_200', 'EMA_200', 'EMA_Fast', 'EMA_Slow']
    mm_features = ['RSI', 'MACD', 'Signal', 'log_returns',
                   'rolling_volatility', 'momentum', 'days_since_traded']
    required_features = ss_features + mm_features + ['volume']

    # Ensure required features are in the dataframe
    missing_features = set(required_features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features for scaling: {missing_features}")

    # Check and log non-numeric data
    for feature in required_features:
        if df[feature].dtype not in ['float64', 'int64']:
            raise ValueError(
                f"Feature '{feature}' must be numeric, found {df[feature].dtype}.")

    # Optionally handle extreme values here if needed
    # Set 'log_returns' to zero if necessary
    df['log_returns'] = 0.0

    # Scaling features
    df.loc[:, ss_features] = ss.transform(df[ss_features])  # StandardScaler
    df[mm_features] = mm.transform(df[mm_features].astype(float))
    df['volume'] = np.log1p(df['volume'])  # Log scale for volume
    df['volume'] = vss.transform(df[['volume']])  # Volume-specific scaler

    logger.info(f"Preprocessed DataFrame: {df.head()}")

    # Convert to tensor (make sure the DataFrame is purely numeric at this point)
    tensor = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0)
    logger.info("Input preprocessing complete")

    return tensor


def make_and_save_prediction(ticker, dataframe):
    model = load_model(ticker=ticker)
    ss = load_ss(ticker=ticker)
    mm = load_mm(ticker=ticker)
    vss = load_vss(ticker=ticker)
    css = load_css(ticker=ticker)
    input_tensor = preprocess_input(df=dataframe, ss=ss, mm=mm, vss=vss)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.item()  # Convert output to a float
        prediction = np.array([prediction])  # Convert float to numpy array
        prediction_original_scale = css.inverse_transform(
            prediction.reshape(-1, 1))

    prediction_original_scale = prediction_original_scale.item()
    save_prediction(ticker, prediction_original_scale)
    return prediction_original_scale


def save_prediction(ticker, new_prediction):
    save_path = f'predictions/{ticker}_predictions.csv'
    save_data = load_from_s3(BUCKET_NAME, save_path)
    existing_df = pd.read_csv(BytesIO(save_data))

    if existing_df.empty or len(existing_df) == 0:
        existing_df = pd.DataFrame(columns=['time_stamp', 'prediction'])

    next_day = datetime.now()
    if next_day.weekday() == 4:
        next_day = next_day + timedelta(days=3)
    else:
        next_day = next_day + timedelta(days=1)
    formatted_timestamp = next_day.strftime('%Y-%m-%d')

    new_prediction_df = pd.DataFrame({
        'time_stamp': [formatted_timestamp],  # Add timestamp
        'prediction': [new_prediction]
    })

    updated_df = pd.concat([existing_df, new_prediction_df], ignore_index=True)

    csv_buffer = StringIO()
    updated_df.to_csv(csv_buffer, index=False)

    s3.put_object(Bucket=BUCKET_NAME, Key=save_path,
                  Body=csv_buffer.getvalue())


def save_data(df, s3_key):
    buffer = BytesIO()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df_sorted = df.sort_values(by='time_stamp', ascending=False)
    csv = df_sorted.to_csv(buffer, mode='a', index=False)
    buffer.seek(0)
    s3.upload_fileobj(buffer, BUCKET_NAME, s3_key)


def fetch_and_save_data(ticker):
    api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
    if not api_key:
        return {
            'statusCode': 500,
            'body': 'API key not found in environment variables.'
        }
    
    # Load existing data from S3
    s3_key = f"data/{ticker}_daily_data.csv"
    try:
        existing_data = load_from_s3(BUCKET_NAME, s3_key)
        df_existing = pd.read_csv(BytesIO(existing_data))
    except Exception as e:
        logger.info(f"Error loading data from S3: {e}")
        df_existing = pd.DataFrame()  # Start with an empty DataFrame if the file doesn't exist

    # Fetch new data from Alpha Vantage
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        time_series = data.get('Time Series (Daily)', {})
        
        # Prepare new data
        new_rows = []
        for date, daily_data in time_series.items():
            row = {
                'time_stamp': date,
                'open': daily_data.get('1. open', ''),
                'high': daily_data.get('2. high', ''),
                'low': daily_data.get('3. low', ''),
                'close': daily_data.get('4. close', ''),
                'volume': daily_data.get('5. volume', '')
            }
            new_rows.append(row)
        
        # Create DataFrame from new rows
        df_new = pd.DataFrame(new_rows)
        
        # Combine existing and new data
        if not df_existing.empty:
            df_existing.loc[:, ~df_existing.columns.str.contains('^Unnamed')]
            df_combined = pd.concat([df_existing, df_new])
        else:
            df_combined = df_new
        
        # Remove duplicates based on 'time_stamp'
        df_combined = df_combined.drop_duplicates(subset=['time_stamp'])
        df_sorted = df_combined.sort_values('time_stamp', ascending=False)
        
        # Save the updated DataFrame back to S3
        save_data(df_combined, s3_key)
        logger.info(f"Data for {ticker} saved to {s3_key}")
    else:
        logger.info(f"Failed to fetch data for {ticker}")



def apply_splits(ticker, splits):
    file_path = f'data/{ticker}_daily_data.csv'
    file = load_from_s3(bucket_name=BUCKET_NAME, s3_path=file_path)
    file_bytes = BytesIO(file)
    df = pd.read_csv(file_bytes)

    df['time_stamp'] = pd.to_datetime(df['time_stamp'])

    for split_date, ratio in sorted(splits[ticker].items(), key=lambda x: pd.to_datetime(x[0])):
        split_date = pd.to_datetime(split_date)
        df.loc[df['time_stamp'] <= split_date, [
            'open', 'high', 'low', 'close']] /= ratio
    df['open'] = df['open'].round(2)
    df['high'] = df['high'].round(2)
    df['low'] = df['low'].round(2)
    df['close'] = df['close'].round(2)

    return df


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
    df['rolling_volatility'] = df['log_returns'].rolling(
        window=window).std() * np.sqrt(252)


def calculate_momentum(df, n=10):
    """Calculate momentum."""
    df['momentum'] = df['close'] - df['close'].shift(n)


def convert_date(df):
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])

    # Calculate the days since the ticker started trading
    df['days_since_traded'] = (
        df['time_stamp'] - df['time_stamp'].min()).dt.days

    return df


def engineer_features(df):
    df = reverse_dataframe(df)
    calculate_moving_averages(df, [10, 20, 50, 100, 200])
    relative_strength_index(df)
    calculate_macd(df)
    calculate_bollinger_bands(df)
    calculate_historical_volatility(df)
    calculate_momentum(df)
    convert_date(df)
    df = reverse_dataframe(df)
    return df


def main_pipeline(ticker):
    data_file_path = f'data/{ticker}_daily_data.csv'
    fetch_and_save_data(ticker=ticker)
    data_df = apply_splits(ticker=ticker, splits=SPLITS)
    features_df = engineer_features(df=data_df)
    save_data(features_df, data_file_path)
    make_and_save_prediction(ticker=ticker, dataframe=features_df)


def lambda_handler(event, context):
    tickers = ['AAPL']
    for ticker in tickers:
        main_pipeline(ticker)

    return {
        'statusCode': 200,
        'body': json.dumps('Data fetched and saved. Predicition added and saved to S3')
    }
