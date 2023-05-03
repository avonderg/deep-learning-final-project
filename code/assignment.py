import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from preprocess import get_data
from lstm import LSTM
from gru import GRU

# Ensures that we run only on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Global variables
window_size = 60
test_samples = 251
future_time_steps = 20


def train_model(model, input_data, target_data, learning_rate=0.0009, num_epochs=100):
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(input_data)

        output = output.view(-1, model.fc_final.out_features)
        target = target_data.view(-1, model.fc_final.out_features)

        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

def statistical_tests(actual: np.ndarray, predicted: np.ndarray):
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mda = np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))
    return rmse, mda

# def rescaleit(y, mima):
#     yt = (y * (mima[1] - mima[0])) + mima[0]
#     return yt

# Repeats the last known value f times
def baselinef(U, f):
    last = U.shape[0]
    yhat = np.zeros((last, f))
    for j in range(0, last):
        yhat[j, 0:f] = np.repeat(U[j, U.shape[1] - 1], f)
    return yhat

def rescaleit(y, mima):
    yt = (y * (mima[1] - mima[0])) + mima[0]
    return yt

def plot_predictions(testY, test_predictions, baseline_predictions, fig_num, min_max, sample_index):
    testY_rescaled = rescaleit(testY[sample_index], min_max[0])
    test_predictions_rescaled = rescaleit(test_predictions[sample_index], min_max[0])
    baseline_predictions_rescaled = rescaleit(baseline_predictions[sample_index], min_max[0])

    plt.figure(fig_num, figsize=(16, 8))
    plt.plot(testY_rescaled, label='True Future')
    plt.plot(test_predictions_rescaled, label='Predicted Future')
    plt.plot(baseline_predictions_rescaled, label='Baseline Future')
    plt.legend(loc=1)
    plt.title(f'Predictions vs True Future for Sample {sample_index}')
    plt.show()


def initialize_model(use_LSTM, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
    if use_LSTM:
        return LSTM(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob)
    else:
        return GRU(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob)

def main(use_LSTM=True, use_activities_dataset=True):
    model_name = "LSTM" if use_LSTM else "GRU"
    dataset_name = "activities.csv" if use_activities_dataset else "BANKEX.csv"
    print(f'Running {model_name} model on {dataset_name} dataset.\n')

    window_size = 50
    test_samples = 100
    future_time_steps = 10
    input_dim = 1
    hidden_dim = 256
    output_window = 10
    dropout_prob = 0.2
    layer_dim = 2
    output_dim = 10

    trainX, trainY, testX, testY, X, min_max, num_samples = get_data(window_size, test_samples, future_time_steps, use_activities_dataset=use_activities_dataset)

    trainX_tensor = torch.tensor(trainX).float().view(-1, window_size, input_dim)
    trainY_tensor = torch.tensor(trainY).float()
    testX_tensor = torch.tensor(testX).float().view(-1, window_size, input_dim)
    testY_tensor = torch.tensor(testY).float()

    model = initialize_model(use_LSTM, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob)

    # Train the model
    train_model(model, trainX_tensor, trainY_tensor, learning_rate=0.0009, num_epochs=100)

    # Test the model
    with torch.no_grad():
        test_predictions = model(testX_tensor)

    # Baseline predictions
    baseline_predictions = baselinef(testX, future_time_steps)

    # Calculate the mean squared error
    mse = F.mse_loss(test_predictions, testY_tensor).item()
    print(f"Mean squared error on the test set: {mse:.4f}")

    # Convert the predictions and ground truth to NumPy arrays
    test_predictions_np = test_predictions.numpy()
    testY_np = testY_tensor.numpy()

    # Calculate RMSE and MDA
    rmse, mda = statistical_tests(testY_np, test_predictions_np)
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Directional Accuracy: {mda:.4f}")

    # Plot the predictions, ground truth, and baseline
    for i in range(9):
        plot_predictions(testY_np, test_predictions_np, baseline_predictions, i, min_max, i)

if __name__ == '__main__':
    main(use_LSTM=False, use_activities_dataset=True)  # Set this flag to False if you want to use the GRU model

