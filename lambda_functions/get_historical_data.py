import boto3
import pandas as pd
from io import BytesIO
import json

s3 = boto3.client('s3')
tickers = []
bucket_name = 'faangfinance'


def get_predictions(ticker):
    try:
        file_key = f'predictions/{ticker}_predictions.csv'
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        data = obj['Body'].read()

        df = pd.read_csv(BytesIO(data))
        return df
    except Exception as e:
        print(f"Error fetching prediction for {ticker}: {e}")
        return None


def get_close_prices(ticker):
    try:
        predictions_df = get_predictions(ticker)
        file_key = f'data/{ticker}_daily_data.csv'
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        data = obj['Body'].read()

        df = pd.read_csv(BytesIO(data))
        return df
    except Exception as e:
        print(f"Error fetching previous close for {ticker}: {e}")
        return None


def merge(ticker):
    predictions_df = get_predictions(ticker)
    close_prices = get_close_prices(ticker)

    result = pd.merge(
        predictions_df, df[['time_stamp', 'close']], on='time_stamp', how='left')
    return result


def lambda_handler(event, context):

    tickers = ['AAPL', 'GOOG', 'META', 'NFLX', 'AMZN', 'NVDA', 'MSFT', 'ADBE']

    path_parameters = event.get("pathParameters")
    if path_parameters is not None:
        ticker = path_parameters.get('ticker')

    else:
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid Request, no path parameter recognized')
        }
    if ticker in tickers:
        data = {}
        ticker = event['pathParameters']['ticker']
        df = merge(ticker)
        df.sort_values('time_stamp', inplace=True)
        for index, row in df.iterrows():
            data[row['time_stamp']] = {
                'prediction': row['prediction'],
                'actual': row['close']
            }

        json_data = json.dumps(data)

        return {
            'statusCode': 200,
            'body': json_data
        }

    else:
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid ticker')
        }
