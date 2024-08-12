from flask import Flask, request, jsonify
import torch
import numpy as np
from lstm_model import LSTMModel 

app = Flask(__name__)


def load_model(ticker): 
    model_path = f'./models/{ticker}_Model.pth'
    model = LSTMModel(num_features=24, hidden_dim=50, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_input(data):
   
    return torch.tensor(data, dtype=torch.float32).unsqueeze(0)

@app.route('/predict/<ticker>', methods=['POST'])
def predict(ticker):
    data = request.get_json() 
    input_tensor = preprocess_input(data['sequence'])
    model = load_model(ticker)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.item()
    return jsonify({f'{ticker} rediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)