import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import datetime
import json
from tqdm import tqdm
import matplotlib


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LSTMModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(num_features, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) 
        return out
    
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_evaluate_model(model, train_loader, test_loader, epochs, config):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    early_stopping = EarlyStopping(patience=config['patience'], min_delta=config['min_delta'])
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0

        train_iterator = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', total=len(train_loader))
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_train_loss = total_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1}, Average Training Loss: {average_train_loss}')

        if epoch % 10 == 0:  # Evaluation every 10 epochs
            model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                    total_test_loss += loss.item()
            average_test_loss = total_test_loss / len(test_loader)
            logging.info(f'Epoch {epoch+1}, Test Loss: {average_test_loss}')

            early_stopping(average_test_loss)
            if early_stopping.early_stop:
                logging.info("Early stopping triggered")
                break

            model.train()

    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def final_evaluation(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    total_test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc='Final Evaluation', total=len(test_loader)):
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            total_test_loss += loss.item()

    average_test_loss = total_test_loss / len(test_loader)
    print(f'Final Test Loss: {average_test_loss}')
    logging.info(f'Final Test Loss: {average_test_loss}')

def load_config(filepath='config/appl_model_config.json'):
    with open(filepath, 'r') as file:
        config = json.load(file)
    return config

def main(ticker):
    logging.info("Starting training...")    
    config = load_config()
    logging.info(f"Configuration loaded: {config}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train_sequences = torch.load(f'data/{ticker}_sequences/X_train_sequences.pt').to(device)
    y_train_sequences = torch.load(f'data/{ticker}_sequences/y_train_sequences.pt').to(device)
    X_test_sequences = torch.load(f'data/{ticker}_sequences/X_test_sequences.pt').to(device)
    y_test_sequences = torch.load(f'data/{ticker}_sequences/y_test_sequences.pt').to(device)
    
    train_data = TensorDataset(X_train_sequences.to(device), y_train_sequences)
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    test_data = TensorDataset(X_test_sequences, y_test_sequences)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False)

    num_features = X_train_sequences.shape[2]
    model = LSTMModel(num_features, hidden_dim=config['hidden_dim'], num_layers=config['num_layers'], output_size=1)
    model.to(device)

    model = train_evaluate_model(model, train_loader, test_loader, epochs=config['epochs'], config=config)

    logging.info("Model training completed.")
    logging.info("Starting final evaluation...")
    final_evaluation(model, test_loader)
    
    save_model(model, f'model/{ticker}_model_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')

if __name__ == '__main__':
    main()


