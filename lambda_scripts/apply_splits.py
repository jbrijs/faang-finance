import pandas as pd
import boto3

s3 = boto3.client('s3')
BUCKET_NAME = 'faangfinance'

def load_from_s3(bucket_name, s3_path):
    obj = s3.get_object(Bucket=bucket_name, Key=s3_path)
    return obj['Body'].read()


def apply_splits(ticker):
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


splits = {
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
