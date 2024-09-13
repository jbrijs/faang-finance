from flask import Flask, request, jsonify
import torch
import numpy as np
from lstm_model import LSTMModel 
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import boto3
from io import BytesIO

app = Flask(__name__)
s3 = boto3.client('s3')
BUCKET_NAME = 'faangfinance'

def load_from_s3(bucket_name, s3_path):
    obj = s3.get_object(Bucket=bucket_name, Key=s3_path)
    return obj['Body'].read()


def load_model(ticker): 
    model_path = f'./models/{ticker}_Model.pth'
    model_data = load_from_s3(BUCKET_NAME, model_path)
    model = LSTMModel(num_features=26, hidden_dim=25, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(BytesIO(model_data)))
    model.eval()
    return model

def load_scaler(scaler_type, ticker):
    scaler_path = f'./data/{ticker}_scalers/{scaler_type}.pkl'
    scaler_data = load_from_s3(BUCKET_NAME, scaler_path)
    return joblib.load(BytesIO(scaler_data))

def load_ss(ticker):
    return load_scaler('ss', ticker)

def load_mm(ticker):
    return load_scaler('mm', ticker)

def load_vss(ticker):
    return load_scaler('vss', ticker)
    
def load_css(ticker):
    return load_scaler('css', ticker)

def prepare_data(ticker):
    data_path = f'./data/{ticker}_daily_data.csv'
    data = load_from_s3(BUCKET_NAME, data_path)
    df = pd.read_csv(BytesIO(data))
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df.sort_values('time_stamp', inplace=True)
    df['days_since_traded'] = (df['time_stamp'] - df['time_stamp'].min()).dt.days
    df = df.dropna()
    return df.iloc[-10:]

def preprocess_input(df, ss, mm, vss, css):
    df = df.drop('time_stamp', axis=1)
    df['volume'] = df['volume'].astype(float)

    ss_features = ['open', 'high', 'low', 'close', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 'SMA_100', 'EMA_100', 'SMA_200', 'EMA_200', 'EMA_Fast', 'EMA_Slow']
    mm_features = ['RSI', 'MACD', 'Signal', 'log_returns', 'rolling_volatility', 'momentum', 'days_since_traded']

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
