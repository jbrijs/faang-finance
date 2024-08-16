from flask import Flask, request, jsonify
import torch
import numpy as np
from lstm_model import LSTMModel 
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

app = Flask(__name__)


def load_model(ticker): 
    model_path = f'./models/{ticker}_Model.pth'
    model = LSTMModel(num_features=24, hidden_dim=50, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_ss(ticker):
    scaler_path = f'./data/{ticker}_scalars/ss.pkl'
    ss = joblib.load(scaler_path)
    return ss

def load_mm(ticker):
    scaler_path = f'data/{ticker}_scalars/mm.pkl'
    mm = joblib.load(scaler_path)
    return mm

def load_vss(ticker):
    scaler_path = f'data/{ticker}_scalars/vss.pkl'
    vss = joblib.load(scaler_path)
    return vss

def prepare_data(ticker):
    filepath = f'./data/{ticker}_daily_data.csv'
    df = pd.read_csv(filepath)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df.sort_values('time_stamp', inplace=True)
    df = df.dropna()
    return df.iloc[:10]

def preprocess_input(df, ss, mm, vss):
    # Dropping 'time_stamp' column and ensuring 'volume' is a float
    df = df.drop('time_stamp', axis=1)
    df['volume'] = df['volume'].astype(float)

    # Define features for scaling
    ss_features = ['open', 'high', 'low', 'close', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 'SMA_100', 'EMA_100', 'SMA_200', 'EMA_200', 'EMA_Fast', 'EMA_Slow']
    mm_features = ['RSI', 'MACD', 'Signal', 'log_returns', 'rolling_volatility', 'momentum']

    # Apply transformations
    if any(feat not in df for feat in ss_features + mm_features):
        raise ValueError("DataFrame lacks required features for scaling")

    df[ss_features] = ss.transform(df[ss_features])
    df[mm_features] = mm.transform(df[mm_features])
    df['volume'] = np.log1p(df['volume'])
    df['volume'] = vss.transform(df[['volume']])

    # Convert to tensor, ensuring all data is numeric and no NaN values exist
    if df.isnull().any().any():
        raise ValueError("NaN values found in DataFrame after processing")
    
    tensor = torch.tensor(df.values, dtype=torch.float32)
    return tensor

@app.route('/predict/<ticker>', methods=['GET'])
def predict(ticker):
    model = load_model(ticker=ticker)
    ss = load_ss(ticker=ticker)
    mm = load_mm(ticker=ticker)
    vss = load_vss(ticker=ticker)
    dataframe = prepare_data(ticker=ticker)
    input_tensor = preprocess_input(df=dataframe)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.item()
    return jsonify({f'{ticker} prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
