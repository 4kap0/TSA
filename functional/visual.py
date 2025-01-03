import numpy as np
import matplotlib.pyplot as plt

def pred_and_loss(pred_inverse, Y_inverse, MAE_score, train_hists, days):
    days = days

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,5))
    ax1.plot(np.arange(len(pred_inverse)), pred_inverse, label = 'pred')
    ax1.plot(np.arange(len(Y_inverse)), Y_inverse, label = 'true')
    ax1.set_xticks(list(np.linspace(0, len(Y_inverse), 6, dtype = int)), labels = days)
    ax1.set(xlabel = 'Day', ylabel = 'Price')
    ax1.set_title("Prediction Plot")
    ax1.legend()

    train_hist_selected = train_hists[np.argmin(MAE_score)]

    # Loss per Epoch
    ax2.plot(train_hist_selected, label="Training loss")
    ax2.vlines(np.argmin(train_hist_selected), ymin = -0.001, ymax= max(train_hist_selected) / 2,
                colors = 'OrangeRed', label = 'Early Stop')
    ax2.set(xlabel = 'Epoch', ylabel = 'Avg_Loss')
    ax2.set_title('Training loss')
    ax2.legend()