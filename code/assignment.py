from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from lstm import LSTM
# from convolution import conv2d

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


def train_lstm(model, input_data, target_data, learning_rate=0.001, num_epochs=100):
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



def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 
    
    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.
    
    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.
    
    :return: None
    '''

    # Assuming the get_data function returns trainX, trainY, testX, testY, X, and min_max
    window_size = 50
    test_samples = 100
    future_time_steps = 10

    # Create an instance of the LSTMModel
    input_dim = 1
    hidden_dim = 128
    output_window = 10

    trainX, trainY, testX, testY, X, min_max = get_data(window_size, test_samples, future_time_steps)

    # Convert the numpy arrays to PyTorch tensors and reshape them accordingly
    trainX_tensor = torch.tensor(trainX).float().view(-1, window_size, input_dim)  # (batch_size, sequence_length, input_dim)
    trainY_tensor = torch.tensor(trainY).float()  # (batch_size, output_dim)
    testX_tensor = torch.tensor(testX).float().view(-1, window_size, input_dim)  # (batch_size, sequence_length, input_dim)
    testY_tensor = torch.tensor(testY).float()  # (batch_size, output_dim)


    model = LSTM(input_dim, hidden_dim, output_window)

    # Train the model
    train_lstm(model, trainX_tensor, trainY_tensor, learning_rate=0.001, num_epochs=100)



    # model = Model()

    # for epoch in range(model.epochs):
    #     train(model, train_inputs, train_labels)
    # # visualize_loss(model.loss_list)
    # accuracy = test(model,test_inputs,test_labels)
    # print(accuracy)



if __name__ == '__main__':
    main()
