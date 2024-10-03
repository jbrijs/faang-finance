import boto3
import pandas as pd
from io import BytesIO
import json
import logging
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)
s3 = boto3.client('s3')
bucket_name = 'faangfinance'


def get_predictions(ticker):
    try:
        file_key = f'predictions/{ticker}_predictions.csv'
        obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        data = obj['Body'].read()
        date = datetime.today().strftime('%Y-%m-%d')

        df = pd.read_csv(BytesIO(data))
        df = df[df['time_stamp' != date]]
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
    close_prices_df = get_close_prices(ticker)

    result = pd.merge(
        predictions_df, close_prices_df[['time_stamp', 'close']], on='time_stamp', how='left')
    return result


def lambda_handler(event, context):
    tickers = ['AAPL', 'GOOG', 'META', 'NFLX', 'AMZN', 'NVDA', 'MSFT', 'ADBE']

    # Get path parameters
    path_parameters = event.get("pathParameters")
    if path_parameters is not None:
        ticker = path_parameters.get('ticker')
    else:
        logger.info("No path param recognized")
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid Request, no path parameter recognized')
        }

    # Check if ticker is valid
    if ticker in tickers:
        ticker = event['pathParameters']['ticker']
        df = merge(ticker)
        df.sort_values('time_stamp', inplace=True)
        data = []

        for index, row in df.iterrows():
            data.append({
                'timeStamp': row['time_stamp'],
                'prediction': row['prediction'],
                'actual': row['close']
            })

        logger.info('Success')
        
        # Properly format the response for API Gateway
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(data), 
            "isBase64Encoded": False
        }

    else:
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid ticker')
        }