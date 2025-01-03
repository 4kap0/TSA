######### Not Completed. Doesn't work well

from functional.model import LSTM_M, train_model, prediction, MAE
from functional.preprocessing import get_price, data_adjust, train_test_split, scailing, build_dataset
from functional.visual import pred_and_loss
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Load data & Preprocessing
print('Load data and Preprocessing ....')

df = get_price('APPL')
df, dates = data_adjust(df)

print(df)

seq_length = 50
batch = 100

# split
train_size = int(len(df)*0.8)
train_set, test_set = train_test_split(df, train_size, seq_length)

# scailing
scaler_x, scaler_y, train_set, test_set = scailing(train_set, test_set)

# Build Time Series Structure for training
trainX, trainY = build_dataset(np.array(train_set), seq_length)
testX  ,testY  = build_dataset(np.array(test_set) , seq_length)

# Tensorization
trainX_tensor = torch.FloatTensor(trainX).to(device)
trainY_tensor = torch.FloatTensor(trainY).to(device)

testX_tensor = torch.FloatTensor(testX).to(device)
testY_tensor = torch.FloatTensor(testY).to(device)

# mk DataLoader
dataset = TensorDataset(trainX_tensor, trainY_tensor)
dataloader = DataLoader(dataset,
                        batch_size=batch,
                        shuffle=True,  
                        drop_last=True)

print('Done.')
# hyperparams
data_dim = 5
hidden_dim = 10 
output_dim = 1 
learning_rate = 0.01
nb_epochs = 500

print('Start Training....')
## Training part
while True:
    MAE_score = []
    train_hists = []
    for num in range(20):

        # Training
        model = LSTM_M(data_dim, hidden_dim, seq_length, output_dim, 1).to(device)  
        trained_model, train_hist = train_model(model, dataloader, num_epochs = nb_epochs, lr = learning_rate, 
                                        verbose = 20, patience = 10)
        train_hists.append(train_hist)

        # Save parameter
        PATH = "./params/model_params_{}/LSTM_{}_days_prediction{}.pth".format(seq_length, seq_length, num)
        torch.save(model.state_dict(), PATH)
        
        model.eval()
        
        pred_inverse, Y_inverse = prediction(model, testX_tensor, testY_tensor, scaler_y)

        print('VAL MAE SCORE : ', MAE(pred_inverse, Y_inverse),'\n\n')
        MAE_score.append(MAE(pred_inverse, Y_inverse))

    if min(MAE_score) < 10:
        break

print('Training is done, load best params.')
# Load best parameter
best_params = "./params/model_params_{}/LSTM_{}_days_prediction{}.pth".format(seq_length, seq_length, np.argmin(MAE_score))
model = LSTM_M(data_dim, hidden_dim, seq_length, output_dim, 1).to(device)
model.load_state_dict(torch.load(best_params))
model.eval()

# prediction
pred_inverse, Y_inverse = prediction(model, testX_tensor, testY_tensor, scaler_y)
print('MAE SCORE : ', MAE(pred_inverse, Y_inverse))

# visualization
days = [str(dates[train_size + i - 1])[:10] for i in list(np.linspace(0, len(testY_tensor), 6, dtype = int))]
pred_and_loss(pred_inverse, Y_inverse, MAE_score, train_hists, days)

# save model params    
PATH = "./params/best_params/LSTM_{}_days_prediction.pth".format(seq_length)
torch.save(model.state_dict(), PATH)

# save fig
plt.savefig('./fig/seq_len_{}.png'.format(seq_length), facecolor='#eeeeee')
