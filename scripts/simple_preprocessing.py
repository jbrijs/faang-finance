import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import torch

def prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df.sort_values('time_stamp', inplace=True)
    df = df[['time_stamp', 'open', 'high', 'low', 'volume', 'close']]
    df['next_day_close'] = df['close'].shift(-1)  
    return df.dropna()

def train_test_split(df):
    # Prepare data
    y = df['next_day_close'].values
    X = df.drop(columns=['next_day_close', 'time_stamp']).values  # Ensure dropping correctly

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=20)  # You can adjust n_splits based on your dataset size and needs

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test

def scale(X_train, X_test):
    ss = StandardScaler()
    volume_scaler = StandardScaler()

    # Scale the 'open', 'high', 'low', 'close' features
    X_train[:, :4] = ss.fit_transform(X_train[:, :4])
    X_test[:, :4] = ss.transform(X_test[:, :4])

    # Correctly handle 'volume' as a 2D array for scaling
    X_train[:, 4:5] = volume_scaler.fit_transform(X_train[:, 4:5])
    X_test[:, 4:5] = volume_scaler.transform(X_test[:, 4:5])

    return X_train, X_test


def create_tensors(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor

def create_sequences(input_data, seq_length):
    xs, ys = [], []
    for i in range(len(input_data) - seq_length):
        x = input_data[i:(i + seq_length)]
        y = input_data[i + seq_length]  # Ensure y is correctly pointing to a single value or properly shaped tensor
        xs.append(x)
        ys.append(y[-1])  # Make sure y is a scalar or a single-dimensional entry if it's a tensor

    # Convert lists to tensors
    xs_tensor = torch.stack(xs)
    ys_tensor = torch.tensor(ys, dtype=torch.float32)  # Ensure ys contains appropriate shapes/sizes
    
    # Debug prints to understand the shapes
    print(f"Shape of xs_tensor: {xs_tensor.shape}")
    print(f"Shape of ys_tensor: {ys_tensor.shape}")
    
    return xs_tensor, ys_tensor



def main(ticker):
    print("Preprocessing data and creating tensor sequences...")
    file = f'data/{ticker}_daily_data.csv'

    df = prepare_data(file)
    X_train, X_test, y_train, y_test = train_test_split(df)
    X_train, X_test = scale(X_train, X_test)

    X_train_tensor, y_train_tensor = create_tensors(X_train, y_train)
    X_test_tensor, y_test_tensor = create_tensors(X_test, y_test)
    seq_length = 30

    X_train_sequences, y_train_sequences = create_sequences(X_train_tensor, seq_length)
    X_test_sequences, y_test_sequences = create_sequences(X_test_tensor, seq_length)

    print(f"X_train_sequence shape: {X_train_sequences.shape}")
    print(f"X_test_sequence shape: {X_test_sequences.shape}")
    print(f"y_train_sequence shape: {y_train_sequences.shape}")
    print(f"y_test_sequence shape: {y_test_sequences.shape}")

    # Save tensors
    torch.save(X_train_sequences, f'data/{ticker}_sequences/simple_X_train_sequences.pt')
    torch.save(y_train_sequences, f'data/{ticker}_sequences/simple_y_train_sequences.pt')
    torch.save(X_test_sequences, f'data/{ticker}_sequences/simple_X_test_sequences.pt')
    torch.save(y_test_sequences, f'data/{ticker}_sequences/simple_y_test_sequences.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch data for a specific stock ticker.')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol')
    args = parser.parse_args()
    main(args.ticker)
