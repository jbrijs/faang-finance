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


# def get_secret():
#     secret_name = "faang-finance-secret"
#     region_name = "us-west-1"

#     client = boto3.client("secretsmanager", region_name=region_name)
#     try:
#         response = client.get_secret_value(SecretId=secret_name)
#         secret = response["SecretString"]
#         secret_dict = json.loads(secret)
#         return secret_dict["VANTAGE_API_KEY"]

#     except Exception as e:
#         print(f"Error fetching secret: {e}")
#         raise e


def load_from_s3(bucket_name, s3_path):
    obj = s3.get_object(Bucket=bucket_name, Key=s3_path)
    return obj['Body'].read()


def load_model(ticker):
    model_path = f'models/{ticker}_Model.pth'
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


def prepare_data(ticker):
    data_path = f'data/{ticker}_daily_data.csv'
    data = load_from_s3(BUCKET_NAME, data_path)
    df = pd.read_csv(BytesIO(data))
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df.sort_values('time_stamp', inplace=True)
    df['days_since_traded'] = (
        df['time_stamp'] - df['time_stamp'].min()).dt.days
    df = df.dropna()
    return df.iloc[-10:]


def preprocess_input(df, ss, mm, vss):
    df = df.drop('time_stamp', axis=1)
    df['volume'] = df['volume'].astype(float)

    ss_features = ['open', 'high', 'low', 'close', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20',
                   'SMA_50', 'EMA_50', 'SMA_100', 'EMA_100', 'SMA_200', 'EMA_200', 'EMA_Fast', 'EMA_Slow']
    mm_features = ['RSI', 'MACD', 'Signal', 'log_returns',
                   'rolling_volatility', 'momentum', 'days_since_traded']

    # Ensure the DataFrame only contains the features expected by the scalers
    if set(ss_features + mm_features + ['volume']).issubset(df.columns):
        df.loc[:, ss_features] = ss.transform(df[ss_features])
        df.loc[:, mm_features] = mm.transform(df[mm_features])
        df.loc[:, 'volume'] = np.log1p(df['volume'])
        df.loc[:, 'volume'] = vss.transform(
            df[['volume']].to_numpy().reshape(-1, 1))
    else:
        missing_features = set(ss_features + mm_features +
                               ['volume']) - set(df.columns)
        raise ValueError(f"Missing features for scaling: {missing_features}")

    tensor = torch.tensor(df.values, dtype=torch.float32).unsqueeze(0)
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
    csv = df.to_csv(buffer)
    buffer.seek(0)
    s3.upload_fileobj(buffer, BUCKET_NAME, s3_key)


def fetch_and_save_data(ticker):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        time_series = data.get('Time Series (Daily)', {})
        filename = f"data/{ticker}_daily_data.csv"
        keys = ['time_stamp', 'open', 'high', 'low', 'close', 'volume']
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            for date, daily_data in time_series.items():
                row = {'time_stamp': date,
                       'open': daily_data.get('1. open', ''),
                       'high': daily_data.get('2. high', ''),
                       'low': daily_data.get('3. low', ''),
                       'close': daily_data.get('4. close', ''),
                       'volume': daily_data.get('5. volume', '')}
                writer.writerow(row)
        print(f"Data for {ticker} written to {filename}")
    else:
        print(f"Failed to fetch data for {ticker}")


def apply_splits(ticker, splits):
    file_path = f'data/{ticker}_daily_data.csv'
    file = load_from_s3(bucket_name=BUCKET_NAME, s3_path=file_path)
    df = pd.read_csv(file)

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
    tickers = ['AAPL', 'GOOG', 'META', 'NFLX', 'AMZN', 'NVDA', 'MSFT', 'ADBE']
    for ticker in tickers:
        main_pipeline(ticker)

    return {
        'statusCode': 200,
        'body': json.dumps('Data fetched and saved. Predicition added and saved to S3')
    }
