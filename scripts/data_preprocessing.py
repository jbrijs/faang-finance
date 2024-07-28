import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse

filepath = './processed_data/AAPL_daily_data_splits_processed.csv'

class TrainingData:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    


def prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df.sort_values('time_stamp', inplace=True)
    df = df.dropna()
    return df


def train_test_split(df):
    train_size = int(len(df) * 0.8)
    X = df.drop('close', axis=1)
    y = df['close']
    X_train = X.iloc[:train_size].copy()
    X_test = X.iloc[train_size:].copy()
    y_train = y.iloc[:train_size].copy()
    y_test = y.iloc[train_size:].copy() 

    X_train['volume'] = X_train['volume'].astype(float)
    X_test['volume'] = X_test['volume'].astype(float)

    training_data = TrainingData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    return training_data

def scale(training_data):
    ss = StandardScaler()
    ss_features = ['open', 'high', 'low', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 'SMA_100', 'EMA_100', 'SMA_200', 'EMA_200', 'EMA_Fast', 'EMA_Slow']

    training_data.X_train.loc[:, ss_features] = ss.fit_transform(training_data.X_train[ss_features])
    training_data.X_test.loc[:, ss_features] = ss.transform(training_data.X_test[ss_features])

    mm = MinMaxScaler()
    mm_features = ['RSI', 'MACD', 'Signal', 'log_returns', 'rolling_volatility', 'momentum']

    # Fit on training data and transform both training and testing data
    training_data.X_train.loc[:, mm_features] = mm.fit_transform(training_data.X_train[mm_features])
    training_data.X_test.loc[:, mm_features] = mm.transform(training_data.X_test[mm_features])

    # Ensure 'volume' is float before log transformation and apply log transformation
    training_data.X_train.loc[:, 'volume'] = np.log1p(training_data.X_train['volume'].astype(float))
    training_data.X_test.loc[:, 'volume'] = np.log1p(training_data.X_test['volume'].astype(float))

    # Apply StandardScaler to 'volume'
    volume_scaler = StandardScaler()
    training_data.X_train.loc[:, 'volume'] = volume_scaler.fit_transform(training_data.X_train[['volume']])
    training_data.X_test.loc[:, 'volume'] = volume_scaler.transform(training_data.X_test[['volume']])

    return training_data

def create_tensors(training_data):
    X_train_tensor = torch.tensor(training_data.X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(training_data.X_test.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(training_data.y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(training_data.y_test.values, dtype=torch.float32)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def create_sequences(input_data, seq_length):
    xs = []
    ys = []

    for i in range(len(input_data) - seq_length):
        x = input_data[i:(i+seq_length)] 
        y = input_data[i + seq_length]
        xs.append(x)
        ys.append(y)

    return torch.stack(xs), torch.tensor(ys, dtype=torch.float32)


def main(ticker):
    file = f'../data/{ticker}_daily_data.csv'

    data_frame = prepare_data(file)
    training_data = train_test_split(data_frame)
    scaled_data = scale(training_data)
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = create_tensors(scaled_data)
    seq_length = 10  # Define sequence length
    X_train_sequences, y_train_sequences = create_sequences(X_train_tensor, seq_length)
    X_test_sequences, y_test_sequences = create_sequences(X_test_tensor, seq_length)

    # Save tensors
    torch.save(X_train_sequences, '../data/AAPL_sequences/X_train_sequences.pt')
    torch.save(y_train_sequences, '../data/AAPL_sequences/y_train_sequences.pt')
    torch.save(X_test_sequences, '../data/AAPL_sequences/X_test_sequences.pt')
    torch.save(y_test_sequences, '../data/AAPL_sequences/y_test_sequences.pt')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch data for a specific stock ticker.')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    args = parser.parse_args()

    main(args.ticker)




