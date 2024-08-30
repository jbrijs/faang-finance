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
    model = LSTMModel(num_features=25, hidden_dim=25, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_ss(ticker):
    scaler_path = f'./data/{ticker}_scalers/ss.pkl'
    ss = joblib.load(scaler_path)
    return ss

def load_mm(ticker):
    scaler_path = f'data/{ticker}_scalers/mm.pkl'
    mm = joblib.load(scaler_path)
    return mm

def load_vss(ticker):
    scaler_path = f'data/{ticker}_scalers/vss.pkl'
    vss = joblib.load(scaler_path)
    return vss
    
def load_css(ticker):
    scaler_path = f'data/{ticker}_scalers/css.pkl'
    css = joblib.load(scaler_path)
    return css

def prepare_data(ticker):
    filepath = f'./data/{ticker}_daily_data.csv'
    df = pd.read_csv(filepath)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df.sort_values('time_stamp', inplace=True)
    df = df.dropna()
    return df.iloc[:10]

def preprocess_input(df, ss, mm, vss, css):
    df = df.drop('time_stamp', axis=1)
    df['volume'] = df['volume'].astype(float)

    ss_features = ['open', 'high', 'low', 'close', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 'SMA_100', 'EMA_100', 'SMA_200', 'EMA_200', 'EMA_Fast', 'EMA_Slow']
    mm_features = ['RSI', 'MACD', 'Signal', 'log_returns', 'rolling_volatility', 'momentum']

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



@app.route('/predict/<ticker>', methods=['GET'])
def predict(ticker):
    model = load_model(ticker=ticker)
    ss = load_ss(ticker=ticker)
    mm = load_mm(ticker=ticker)
    vss = load_vss(ticker=ticker)
    css = load_css(ticker=ticker)
    dataframe = prepare_data(ticker=ticker)
    input_tensor = preprocess_input(df=dataframe, ss=ss, mm=mm, vss=vss, css=css)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.item()  # Convert output to a float
        prediction = np.array([prediction])  # Convert float to numpy array
        prediction_original_scale = css.inverse_transform(prediction.reshape(-1, 1))

    # Convert the numpy array back to a Python float for JSON serialization
    prediction_original_scale = prediction_original_scale.item()

    # Return the result as a JSON response
    return jsonify({f'{ticker} prediction': prediction_original_scale, 'dataFrame': dataframe.to_dict()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
