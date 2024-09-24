import csv
import requests


def fetch_and_save_data(ticker, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}"
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