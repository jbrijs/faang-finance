import boto3
import pandas as pd
from io import BytesIO

s3 = boto3.client('s3')
tickers = []
bucket_name = 'faangfinance'


def get_prediction(ticker):
    file_key = f'predictions/{ticker}_predictions.csv'
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    data = obj['Body'].read()

    df = pd.read_csv(BytesIO(data))
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])

    most_recent_row = df.loc[df['time_stamp'].idxmax()]
    return most_recent_row['prediction']

def get_previous_close(ticker):
    file_key = f'data/{ticker}_daily_data.csv'
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    data = obj['Body'].read()

    df = pd.read_csv(BytesIO(data))
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])

    most_recent_row = df.loc[df['time_stamp'].idxmax()]
    return most_recent_row['close']


def lambda_handler(event, context):
    predictions = {}
    tickers = ['AAPL', 'GOOG', 'META', 'NFLX', 'AMZN', 'NVDA', 'MSFT', 'ADBE']
    for ticker in tickers:
        predictions[ticker] = {'prediction': get_prediction(ticker), 'prevClose': get_previous_close(ticker)}
    
    
    json_data = df.to_json(orient='records')

    return {
        'statusCode': 200,
        'body': json_data
    }
