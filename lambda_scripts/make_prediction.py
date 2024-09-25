from datetime import datetime, timedelta
from flask import json
import torch
import numpy as np
from lstm_model import LSTMModel
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import boto3
from io import BytesIO, StringIO
import csv
from fetch_data import *
from apply_splits import *
from engineer_features import *

s3 = boto3.client('s3')
BUCKET_NAME = 'faangfinance'

def get_secret():
    secret_name = "faang-finance-secret"
    region_name = "us-west-1"

    client = boto3.client("secretsmanager", region_name=region_name)
    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret = response["SecretString"]
        secret_dict = json.loads(secret)
        return secret_dict["VANTAGE_API_KEY"]
    
    except Exception as e:
        print(f"Error fetching secret: {e}")
        raise e

def load_from_s3(bucket_name, s3_path):
    obj = s3.get_object(Bucket=bucket_name, Key=s3_path)
    return obj['Body'].read()
    
def load_model(ticker): 
    model_path = f'./models/{ticker}_Model.pth'
    model_data = load_from_s3(BUCKET_NAME, model_path)
    model = LSTMModel(num_features=26, hidden_dim=25, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(BytesIO(model_data)))
    model.eval()
    return model

def load_scaler(scaler_type, ticker):
    scaler_path = f'./data/{ticker}_scalers/{scaler_type}.pkl'
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
    data_path = f'./data/{ticker}_daily_data.csv'
    data = load_from_s3(BUCKET_NAME, data_path)
    df = pd.read_csv(BytesIO(data))
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df.sort_values('time_stamp', inplace=True)
    df['days_since_traded'] = (df['time_stamp'] - df['time_stamp'].min()).dt.days
    df = df.dropna()
    return df.iloc[-10:]

def preprocess_input(df, ss, mm, vss):
    df = df.drop('time_stamp', axis=1)
    df['volume'] = df['volume'].astype(float)

    ss_features = ['open', 'high', 'low', 'close', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 'SMA_100', 'EMA_100', 'SMA_200', 'EMA_200', 'EMA_Fast', 'EMA_Slow']
    mm_features = ['RSI', 'MACD', 'Signal', 'log_returns', 'rolling_volatility', 'momentum', 'days_since_traded']

    # Ensure the DataFrame only contains the features expected by the scalers
    if set(ss_features + mm_features + ['volume']).issubset(df.columns):
        df.loc[:, ss_features] = ss.transform(df[ss_features])
        df.loc[:, mm_features] = mm.transform(df[mm_features])
        df.loc[:, 'volume'] = np.log1p(df['volume'])
        df.loc[:, 'volume'] = vss.transform(df[['volume']].to_numpy().reshape(-1, 1))
    else:
        missing_features = set(ss_features + mm_features + ['volume']) - set(df.columns)
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
        prediction_original_scale = css.inverse_transform(prediction.reshape(-1, 1))

    prediction_original_scale = prediction_original_scale.item()
    save_prediction(ticker, prediction_original_scale)
    return prediction_original_scale

def save_prediction(ticker, new_prediction):
    save_path = f'./predictions/{ticker}_predictions.csv'
    save_data = load_from_s3(BUCKET_NAME, save_path)
    existing_df = pd.read_csv(BytesIO(save_data))

    next_day = datetime.now() + timedelta(days=1)
    formatted_timestamp = next_day.strftime('%Y-%m-%d')

    new_prediction_df = pd.DataFrame({
        'time_stamp': [formatted_timestamp],  # Add timestamp
        'prediction': [new_prediction]
    })

    updated_df = pd.concat([existing_df, new_prediction_df], ignore_index=True)

    csv_buffer = StringIO()
    updated_df.to_csv(csv_buffer, index=False)

    s3.put_object(Bucket=BUCKET_NAME, Key=save_path, Body=csv_buffer.getvalue())

def save_data(df, s3_key):
    buffer = BytesIO()
    csv = df.to_csv(buffer)
    buffer.seek(0)
    s3.upload_fileobj(buffer, BUCKET_NAME, s3_key)

def main_pipeline(ticker):
    data_file_path = f'./data/{ticker}_daily_data.csv'
    api_key = get_secret()
    fetch_and_save_data(ticker, api_key)
    data_df = apply_splits(ticker)
    features_df = engineer_features(data_df)
    save_data(features_df, data_file_path)
    make_and_save_prediction(ticker=ticker, dataframe=features_df)


def lambda_handler(event, context):
    api_key = get_secret()
    tickers = ['AAPL', 'GOOG', 'META', 'NFLX', 'AMZN', 'NVDA', 'MSFT', 'ADBE']
    for ticker in tickers:
        main_pipeline(ticker)

    return {
        'statusCode': 200,
        'body': json.dumps('Predicition added and saved to S3')
    }

