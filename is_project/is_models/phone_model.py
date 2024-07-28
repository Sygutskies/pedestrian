import torch

class PhoneModel(torch.nn.Module):
    
    def __init__(self, input_size):
        super().__init__()
        self.lstm1 = torch.nn.LSTM(input_size=input_size, hidden_size=16, num_layers=1, batch_first=True)
        self.ln1 = torch.nn.LayerNorm(16)
        self.lstm2 = torch.nn.LSTM(input_size=16, hidden_size=16, num_layers=1, batch_first=True)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.d1 = torch.nn.Linear(16,16)
        self.d2 = torch.nn.Linear(16,1)
        self.dropout = torch.nn.Dropout(0.5)

        self.apply(initialize_weights)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.tanh(x)
        x = self.ln1(x)
        x, _ = self.lstm2(x)
        x = self.tanh(x)
        x = x[:, -1, :]
        x = self.d1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.d2(x)
        x = torch.sigmoid(x)
        return x
    
def initialize_weights(model):
    if isinstance(model, torch.nn.Linear) or isinstance(model, torch.nn.LSTM) or isinstance(model, torch.nn.LSTMCell):
        for name, param in model.named_parameters():
            if 'weight' in name:
                torch.nn.init.kaiming_normal_(param.data)
            elif 'bias' in name:
                torch.nn.init.constant_(param.data, 0)