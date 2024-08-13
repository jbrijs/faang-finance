from flask import Flask, request, jsonify
import torch
import numpy as np
from lstm_model import LSTMModel 
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

app = Flask(__name__)

ss = StandardScaler()
mm = MinMaxScaler()

def load_model(ticker): 
    model_path = f'./models/{ticker}_Model.pth'
    model = LSTMModel(num_features=24, hidden_dim=50, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess_input(ticker):
    filepath = f'./data/{ticker}_daily_data.csv'
    df = pd.read_csv(filepath)
    df = df.drop('close', axis=1) 
    data = df.iloc[-10:] 

    ss_features = ['open', 'high', 'low', 'SMA_10', 'EMA_10', 'SMA_20', 'EMA_20', 'SMA_50', 'EMA_50', 'SMA_100', 'EMA_100', 'SMA_200', 'EMA_200', 'EMA_Fast', 'EMA_Slow']
    mm_features = ['RSI', 'MACD', 'Signal', 'log_returns', 'rolling_volatility', 'momentum']

    
    data[ss_features] = ss.fit_transform(data[ss_features])
    data[mm_features] = mm.fit_transform(data[mm_features])
    data['volume'] = np.log1p(data['volume'].astype(float)) 
    data['volume'] = ss.transform(data[['volume']])  

    tensor = torch.tensor(data.values, dtype=torch.float32).unsqueeze(0)
    return tensor


@app.route('/predict/<ticker>', methods=['GET'])
def predict(ticker):
    input_tensor = preprocess_input(ticker)
    model = load_model(ticker)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.item()
    return jsonify({f'{ticker} rediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
