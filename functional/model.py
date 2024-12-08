import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

class LSTM_M(nn.Module):
    '''
    input_dim  : 
    hidden_dim :
    seq_len    :
    output_dim :
    layers     :
    '''
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(LSTM_M, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers,
                            # dropout = 0.1,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 128, bias = True)
        self.fc2 = nn.Linear(128, output_dim, bias = True)
        self.active = nn.Tanh()
        
    def reset_hidden_state(self): 
        self.hidden = (
                torch.zeros(self.layers, self.seq_len, self.hidden_dim),
                torch.zeros(self.layers, self.seq_len, self.hidden_dim))
    
    def forward(self, x):
        x, _status = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        x = self.fc1(x[:, -1])     # (batch_size, hidden_dim) : last hidden state
        x = self.active(x)
        x = self.fc2(x)            # (batch_size, output_dim)
        return x

def train_model(model, train_df, num_epochs = 1000, lr = 0.001, verbose = 10, patience = 10):
    '''
    model      : model object to be trained
    train_df   : trainset construted as a torch.utils.data.DataLoader
    num_epochs : 
    lr         : learning_rate
    verbose    :
    patience   : 
    '''    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    nb_epochs = num_epochs
    
    # epoch마다 loss 저장
    train_hist = np.zeros(nb_epochs)

    for epoch in range(nb_epochs):
        avg_cost = 0
        total_batch = len(train_df)
        
        for batch_idx, samples in enumerate(train_df):
            x_train, y_train = samples
            model.reset_hidden_state()
            
            outputs = model(x_train)
            loss = criterion(outputs, y_train)                    
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            avg_cost += loss/total_batch

        train_hist[epoch] = avg_cost        
        
        if epoch % verbose == 0:
            print('Epoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))
            
        if (epoch % patience == 0) & (epoch >= 100):
            
            if (train_hist[epoch-patience] < train_hist[epoch]):
                print(f' Early Stopping : Epoch {epoch} by patience loss.\n Last train loss : {avg_cost}')
                break

    return model.eval(), train_hist

def prediction(model, tensor_X, tensor_Y, scaler_y):
    with torch.no_grad(): 
        pred = []
        for pr in range(len(tensor_X)):

            model.reset_hidden_state()

            predicted = model(torch.unsqueeze(tensor_X[pr], 0))
            predicted = torch.flatten(predicted).item()
            pred.append(predicted)

        # INVERSE
        pred_inverse = scaler_y.inverse_transform(np.array(pred).reshape(-1, 1))
        Y_inverse = scaler_y.inverse_transform(tensor_Y.cpu())

    return pred_inverse, Y_inverse

def MAE(true, pred):
    return np.mean(np.abs(true-pred))