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
        self.lstm = nn.LSTM(num_features, hidden_dim,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(x.device)
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


def load_model(ticker):
    model_path = f'./models/{ticker}_model.pth'
    model = LSTMModel(num_features=26, hidden_dim=25,
                      num_layers=2, output_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_scaler(scaler_type, ticker):
    scaler_path = f'./data/{ticker}_scalers/{scaler_type}.pkl'
    return joblib.load(scaler_path)


def load_ss(ticker):
    return load_scaler('ss', ticker)


def load_mm(ticker):
    return load_scaler('mm', ticker)


def load_vss(ticker):
    return load_scaler('vss', ticker)


def load_css(ticker):
    return load_scaler('css', ticker)


def preprocess_input(df, ss, mm, vss):
    print("Starting input preprocessing")

    # Drop 'time_stamp' and reset index
    df = df.loc[:10]
    print(f"Original DataFrame: {df.head()}")
    print(f"Shape of DataFrame before tensor conversion: {df.shape}")
    df = df.drop('time_stamp', axis=1)
    print(f"Shape of DataFrame after date drop: {df.shape}")
    print(f"Date drop DataFrame: {df.head()}")

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


def make_and_save_prediction(ticker):
    model = load_model(ticker=ticker)
    ss = load_ss(ticker=ticker)
    mm = load_mm(ticker=ticker)
    vss = load_vss(ticker=ticker)
    css = load_css(ticker=ticker)
    data_path = f'./data/{ticker}_daily_data_test.csv'
    df = pd.read_csv(data_path)
    input_tensor = preprocess_input(df=df, ss=ss, mm=mm, vss=vss)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.item()  # Convert output to a float
        prediction = np.array([prediction])  # Convert float to numpy array
        prediction_original_scale = css.inverse_transform(
            prediction.reshape(-1, 1))

    prediction_original_scale = prediction_original_scale.item()
    return prediction_original_scale


def main_pipeline(ticker):
    print(make_and_save_prediction(ticker=ticker))


# def lambda_handler(event, context):
#     tickers = ['AAPL']
#     for ticker in tickers:
#         main_pipeline(ticker)

#     return {
#         'statusCode': 200,
#         'body': json.dumps('Data fetched and saved. Predicition added and saved to S3')
#     }

if __name__ == "__main__":
    main_pipeline('AAPL')
