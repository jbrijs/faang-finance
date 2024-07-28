import requests
import csv
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime

def fetch_partial_sentiment_data(ticker, time_from, api_key):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={api_key}&limit=1000&time_from={time_from}"
    response = requests.get(url)
    print(f"response status code: {response.status_code}")
    if response.status_code == 200:
        try:
            data = response.json()
            print(f"Items: {data.get('items')}")
            feed = data.get("feed")
            return feed 
        except ValueError:
            print("Error parsing JSON response.")
            return None
    else:
        print(f"Failed to fetch data for {ticker}")
        return None

def write_sentiment_data(ticker, feed, writer):
    if feed is None:
        return None

    for article in feed:
        ticker_sentiment_score = None
        ticker_sentiment_label = None
        ticker_sentiment = article.get('ticker_sentiment', [])
        for ts in ticker_sentiment:
            if ts.get('ticker') == ticker:
                ticker_sentiment_score = ts.get('ticker_sentiment_score')
                ticker_sentiment_label = ts.get('ticker_sentiment_label')
        date = article.get('time_published')
        year = date[:4]
        month = date[4:6]
        day = date[6:8]
        formatted_date = f"{year}-{month}-{day}"
        row = {
            'time-stamp': formatted_date,
            'time_published': date,
            'overall_sentiment_score': article.get('overall_sentiment_score'),
            'overall_sentiment_label': article.get('overall_sentiment_label'),
            'ticker_sentiment_score': ticker_sentiment_score,
            'ticker_sentiment_label': ticker_sentiment_label
        }
        writer.writerow(row)
    return date[:-2]

def write_header(csvfile, keys):
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader
        return writer

def main():
    ticker_date_map = {
        "AAPL": "19991101T0001"
    }
    load_dotenv()
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

    if ALPHA_VANTAGE_API_KEY is None:
        print("API key not found. Please set the ALPHA_VANTAGE_API_KEY environment variable.")
        return

    tickers = ['AAPL']
    keys = ['time-stamp', 'time_published', 'overall_sentiment_score', 'overall_sentiment_label', 'ticker_sentiment_score', 'ticker_sentiment_label']

    if not os.path.exists('./sentiment_data/'):
        os.makedirs('./sentiment_data/')

    for ticker in tickers:
        todays_date = datetime.now().strftime('%Y%m%dT%H%M%S')
        filename = f"./sentiment_data/{ticker}_sentiment_data_{todays_date}.csv"
        time_from = ticker_date_map.get(ticker)
        count = 0

        with open(filename, 'w', newline='') as csvfile:
            writer = write_header(csvfile=csvfile, keys=keys)
        

            while time_from != todays_date and count < 1:
                feed = fetch_partial_sentiment_data(ticker, time_from, ALPHA_VANTAGE_API_KEY)
                print(f"Initial Time From {time_from}")
                time_from = write_sentiment_data(ticker=ticker, feed=feed, writer=writer)
                if time_from is None:
                    print(f"No new data to fetch for {ticker}.")
                    break
                print(f"Changed Time From {time_from}")
                count+=1
                print(f"Count: {count}")

if __name__ == '__main__':
    main()
