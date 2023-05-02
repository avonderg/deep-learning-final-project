from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from lstm import LSTM
from gru import GRU  # Make sure to import your GRU model
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# global vars
window_size = 60
test_samples = 251
future_time_steps = 20


def train_model(model, input_data, target_data, learning_rate=0.001, num_epochs=100):
    # Define the loss function and the optimizer
    loss_function = nn.MSELoss()  # Mean squared error loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Clear the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(input_data)

        # Reshape output and target tensors for loss calculation
        output = output.view(-1, model.fc.out_features)  # (batch_size * sequence_length, tagset_size)
        target = target_data.view(-1, model.fc.out_features)  # (batch_size * sequence_length, tagset_size)

        # Calculate the loss
        loss = loss_function(output, target)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Print the loss for this epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

def plot_predictions(testY, test_predictions, index_plotted):
    plt.figure(figsize=(15, 6))
    plt.plot(testY[:,index_plotted], label='Ground Truth', color='blue')
    plt.plot(test_predictions[:,index_plotted], label='Predicted', color='red')
    plt.xlabel("Days")
    plt.ylabel("Scaled Closing Price")
    plt.legend()
    plt.show()


def main(use_LSTM=True):
    window_size = 50
    test_samples = 100
    future_time_steps = 10

    input_dim = 1
    hidden_dim = 128
    output_window = 10
    dropout_prob = 0.2
    layer_dim = 2  # Number of GRU layers
    output_dim = 10  # Number of output time steps for your time series forecasting task

    output_activation = None  # You can set this to any activation function like torch.sigmoid, torch.tanh, etc.

    trainX, trainY, testX, testY, X, min_max = get_data(window_size, test_samples, future_time_steps)

    trainX_tensor = torch.tensor(trainX).float().view(-1, window_size, input_dim)
    trainY_tensor = torch.tensor(trainY).float()
    testX_tensor = torch.tensor(testX).float().view(-1, window_size, input_dim)
    testY_tensor = torch.tensor(testY).float()

    if use_LSTM:
        model = LSTM(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob)
    else:
        model = GRU(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob)

    # Train the model
    train_model(model, trainX_tensor, trainY_tensor, learning_rate=0.001, num_epochs=100)

    # Test the model
    with torch.no_grad():
        test_predictions = model(testX_tensor)

    # Calculate the mean squared error
    mse = F.mse_loss(test_predictions, testY_tensor).item()
    print(f"Mean squared error on the test set: {mse:.4f}")

    # Convert the predictions and ground truth to NumPy arrays
    test_predictions_np = test_predictions.numpy()
    testY_np = testY_tensor.numpy()

    # Plot the predictions and ground truth
    for i in range(9):
        plot_predictions(testY_np, test_predictions_np,i)



if __name__ == '__main__':
    main(use_LSTM=True)  # Set this flag to False if you want to use the GRU model
