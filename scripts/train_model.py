from sklearn.model_selection import TimeSeriesSplit
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import datetime
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LSTMModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, output_size=1, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(num_features, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout_prob)
        
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.dropout(out[:, -1, :])
        
        out = self.fc(out)
        
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

def train_evaluate_model(model, train_loader, val_loader, config):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    early_stopping = EarlyStopping(patience=config['patience'], min_delta=config['min_delta'])
    
    training_losses = []
    validation_losses = []
    best_val_loss = float('inf')

    for epoch in range(config['epochs']):
        model.train()
        total_train_loss = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)
        training_losses.append(average_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                val_loss = criterion(outputs.squeeze(), y_batch)
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(average_val_loss)

        logging.info(f'Epoch {epoch+1}, Training Loss: {average_train_loss}, Validation Loss: {average_val_loss}')

        # Save the model if the validation loss has decreased
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            save_model(model, f"./models/{config['ticker']}_model.pth")
            logging.info(f"Model saved with validation loss: {best_val_loss}")

        early_stopping(average_val_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping triggered")
            break

    return model, training_losses, validation_losses



def cross_validate(data_sequences, data_labels, config):
    tscv = TimeSeriesSplit(n_splits=5)
    overall_val_scores = []

    for train_idx, val_idx in tscv.split(data_sequences):
        train_data = TensorDataset(data_sequences[train_idx], data_labels[train_idx])
        val_data = TensorDataset(data_sequences[val_idx], data_labels[val_idx])

        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=False)
        val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)

        model = LSTMModel(data_sequences.shape[2], config['hidden_dim'], config['num_layers']).to(config['device'])
        trained_model, train_losses, val_losses = train_evaluate_model(model, train_loader, val_loader, config)

        overall_val_scores.append(val_losses[-1])

    average_val_score = sum(overall_val_scores) / len(overall_val_scores)
    logging.info(f'Average Validation Score: {average_val_score}')
    return average_val_score


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# def final_evaluation(model, test_loader):
#     model.eval()
#     criterion = nn.MSELoss()
#     total_test_loss = 0
#     with torch.no_grad():
#         for X_batch, y_batch in tqdm(test_loader, desc='Final Evaluation', total=len(test_loader)):
#             outputs = model(X_batch)
#             loss = criterion(outputs.squeeze(), y_batch)
#             total_test_loss += loss.item()

#     average_test_loss = total_test_loss / len(test_loader)
#     print(f'Final Test Loss: {average_test_loss}')
#     logging.info(f'Final Test Loss: {average_test_loss}')

def load_config(filepath='config/aapl_model_config.json'):
    with open(filepath, 'r') as file:
        config = json.load(file)
    return config

def plot_learning_curves(training_losses, validation_losses):
    plt.figure(figsize=(10 ,5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('learning_curves.png') 
    # plt.show()  

def main(ticker):
    config = load_config()  # Ensure config has 'device' and 'epochs' defined
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['ticker'] = ticker
    
    X_train_sequences = torch.load(f'data/{ticker}_sequences/X_train_sequences.pt').to(config['device'])
    y_train_sequences = torch.load(f'data/{ticker}_sequences/y_train_sequences.pt').to(config['device'])
    
    average_val_score = cross_validate(X_train_sequences, y_train_sequences, config)

    logging.info(f'Cross-validation completed. Average Validation Score: {average_val_score}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate an LSTM model for given stock ticker')
    parser.add_argument('ticker', type=str, help='Stock ticker symbol (e.g., AAPL, GOOGL)')
    args = parser.parse_args()
    
    main(args.ticker)

# Best val loss: 0.46346

#   epochs: 250,
#   batch_size: 25,
#   hidden_dim: 50,
#   num_layers: 2,
#   learning_rate: 0.0001,
#   patience: 30,
#   min_delta: 0.005

