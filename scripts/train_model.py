import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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

def train_evaluate_model(model, train_loader, test_loader, epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch+1) % 100 == 0:
            print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader)}')

         # Evaluation phase
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            total_test_loss += loss.item()

    average_test_loss = total_test_loss / len(test_loader)
    print(f'Test Loss: {average_test_loss}')

def main(ticker):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train_sequences = torch.load(f'data/{ticker}_sequences/X_train_sequences.pt').to(device)
    y_train_sequences = torch.load(f'data/{ticker}_sequences/y_train_sequences.pt').to(device)
    X_test_sequences = torch.load(f'data/{ticker}_sequences/X_test_sequences.pt').to(device)
    y_test_sequences = torch.load(f'data/{ticker}_sequences/y_test_sequences.pt').to(device)
    
    train_data = TensorDataset(X_train_sequences.to(device), y_test_sequences.to(device))
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_data = TensorDataset(X_test_sequences, y_test_sequences)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    num_features = X_train_sequences.shape[2]
    model = LSTMModel(num_features, hidden_dim=50, num_layers=2, output_size=1)
    model.to(device)


    train_evaluate_model(model, train_loader, test_loader, epochs=500)

if __name__ == '__main__':
    main()


