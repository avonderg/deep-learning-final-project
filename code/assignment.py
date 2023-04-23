from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# global vars
window_size = 60
test_samples = 251
future_time_steps = 20

# class Model(tf.keras.Model):
#     def __init__(self):
#         """
#         This model class will contain the architecture for your CNN that 
#         classifies images. We have left in variables in the constructor
#         for you to fill out, but you are welcome to change them if you'd like.
#         """
#         super(Model, self).__init__()

#         self.batch_size = 256
#         self.num_classes = 2
#         self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

#         # TODO: Initialize all hyperparameters
#         self.epochs = 10
#         self.learning_rate = 0.01
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
#         self.dense_layer_size = 128
        

#         # TODO: Initialize all trainable parameters
#         #tf.Variable on any trainable parameter
#         self.c_filter1 = tf.Variable(tf.random.truncated_normal([5,5,3,16], stddev=0.1)) #convolutional filters- size / channels
#         self.c_bias1 = tf.Variable(tf.zeros(16)) #convolutional biases
#         self.c_filter2 = tf.Variable(tf.random.truncated_normal([5,5,16,20], stddev=0.1)) #convolutional filters- size / channels
#         self.c_bias2 = tf.Variable(tf.zeros(20)) #convolutional biases
#         self.c_filter3 = tf.Variable(tf.random.truncated_normal([3,3,20,20], stddev=0.1)) #convolutional filters- size / channels
#         self.c_bias3 = tf.Variable(tf.zeros(20)) #convolutional biases

#         ########### LAYERS
#         self.layer_weights1 = tf.Variable(tf.random.truncated_normal((80,10), stddev=0.1)) # SPECIFIY SHAPE (shape=..)
#         self.layer_bias1 = tf.Variable(tf.zeros((self.batch_size,10)))
#         self.layer_weights2 = tf.Variable(tf.random.truncated_normal((10,10),stddev=0.1))
#         self.layer_bias2 = tf.Variable(tf.zeros((self.batch_size,10)))
#         self.layer_weights3 = tf.Variable(tf.random.truncated_normal((10,2), stddev=0.1))
#         self.layer_bias3 = tf.Variable(tf.zeros(2))


#     def call(self, inputs, is_testing=False):
#         """
#         Runs a forward pass on an input batch of images.
        
#         :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
#         :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
#         :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
#         """
#         # Remember that
#         # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
#         # shape of filter = (filter_height, filter_width, in_channels, out_channels)
#         # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)


#         # Convolutional layer 1
#         conv1 = tf.nn.conv2d(inputs,self.c_filter1, strides=[1, 2, 2, 1], padding='SAME')
#         conv1 = tf.nn.bias_add(conv1, self.c_bias1)
#         mean, variance = tf.nn.moments(conv1,axes=[0,1,2])
#         # conv1 = tf.nn.batch_normalization(conv1, mean=mean, variance=variance, variance_epsilon=1e-5)
#         conv1 = tf.nn.batch_normalization(conv1, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=1e-5)
#         # RELU 1
#         conv1 = tf.nn.relu(conv1)
#         pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        
#         # Convolutional layer 2
#         conv2 = tf.nn.conv2d(pool1,self.c_filter2, strides=[1, 2, 2, 1], padding='SAME')
#         conv2 = tf.nn.bias_add(conv2, self.c_bias2)
#         mean, variance = tf.nn.moments(conv2,axes=[0,1,2])
#         conv2 = tf.nn.batch_normalization(conv2, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=1e-5)
#         conv2 = tf.nn.relu(conv2)
#         pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#         if is_testing:
#             conv_func = conv2d
#         else:
#             conv_func = tf.nn.conv2d

#         # Convolutional layer 3
#         conv3 = conv_func(pool2, self.c_filter3, strides=[1, 1, 1, 1], padding='SAME')
#         conv3 = tf.nn.bias_add(conv3, self.c_bias3)
#         mean, variance = tf.nn.moments(conv3,axes=[0,1,2])
#         conv3 = tf.nn.batch_normalization(conv3, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=1e-5)
#         conv3 = tf.nn.relu(conv3)
#         conv3 = tf.reshape(conv3, (conv3.shape[0],-1))

#         #  layer 1
#         layer1 = tf.matmul(conv3, self.layer_weights1) + self.layer_bias1
#         layer1 = tf.nn.relu(layer1)
        
#         #dropout 1
#         layer1 = tf.nn.dropout(layer1,rate=0.3)

#         #  layer 2 
#         layer2 = tf.matmul(layer1, self.layer_weights2) + self.layer_bias2

#         #dropout 2
#         layer2 = tf.nn.dropout(layer2,rate=0.3)

#         #  layer 3
#         layer3 = tf.matmul(layer2, self.layer_weights3) + self.layer_bias3
        
#         return layer3
    

#     def loss(self, logits, labels):
#         """
#         Calculates the model cross-entropy loss after one forward pass.
#         Softmax is applied in this function.
        
#         :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
#         containing the result of multiple convolution and feed forward layers
#         :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
#         :return: the loss of the model as a Tensor
#         """
#         return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))

#     def accuracy(self, logits, labels):
#         """
#         Calculates the model's prediction accuracy by comparing
#         logits to correct labels – no need to modify this.
        
#         :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
#         containing the result of multiple convolution and feed forward layers
#         :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

#         NOTE: DO NOT EDIT
        
#         :return: the accuracy of the model as a Tensor
#         """
#         correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
#         return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# def train(model, train_inputs, train_labels):
#     '''
#     Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
#     and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
#     To increase accuracy, you may want to use tf.image.random_flip_left_right on your
#     inputs before doing the forward pass. You should batch your inputs.
    
#     :param model: the initialized model to use for the forward pass and backward pass
#     :param train_inputs: train inputs (all inputs to use for training), 
#     shape (num_inputs, width, height, num_channels)
#     :param train_labels: train labels (all labels to use for training), 
#     shape (num_labels, num_classes)
#     :return: Optionally list of losses per batch to use for visualize_loss
#     '''
#     indices = tf.random.shuffle(tf.range(0,train_inputs.shape[0]))
#     shuffled_input = tf.gather(train_inputs, indices)
#     shuffled_label = tf.gather(train_labels, indices)

#     # shuffled_input = tf.image.random_flip_left_right(shuffled_input) #flipped

#     batch_size = model.batch_size

#     for b, b1 in enumerate(range(batch_size, shuffled_input.shape[0] + 1, batch_size)):
#         b0 = b1 - batch_size
#         batch = shuffled_input[b0:b1]
#         with tf.GradientTape() as tape:
#             logits = model.call(tf.image.random_flip_left_right(batch))
#             loss = model.loss(logits,shuffled_label[b0:b1])
#         grads = tape.gradient(loss, model.trainable_variables)
#         model.optimizer.apply_gradients(zip(grads,model.trainable_variables))
#         model.loss_list.append(loss)

# def test(model, test_inputs, test_labels):
#     """
#     Tests the model on the test inputs and labels. You should NOT randomly 
#     flip images or do any extra preprocessing.
    
#     :param test_inputs: test data (all images to be tested), 
#     shape (num_inputs, width, height, num_channels)
#     :param test_labels: test labels (all corresponding labels),
#     shape (num_labels, num_classes)
#     :return: test accuracy - this should be the average accuracy across
#     all batches
#     """
#     batch_size = model.batch_size  #fix -> taken from the last homework
#     accuracy = []


#     for b, b1 in enumerate(range(batch_size, test_inputs.shape[0] + 1, batch_size)):
#         b0 = b1 - batch_size
#         batch = test_inputs[b0:b1]
#         logits = model.call(batch, True)
#         accuracy.append(model.accuracy(logits, test_labels[b0:b1]))

#     return np.mean(accuracy)


# def visualize_loss(losses): 
#     """
#     Uses Matplotlib to visualize the losses of our model.
#     :param losses: list of loss data stored from train. Can use the model's loss_list 
#     field 

#     NOTE: DO NOT EDIT

#     :return: doesn't return anything, a plot should pop-up 
#     """
#     x = [i for i in range(len(losses))]
#     plt.plot(x, losses)
#     plt.title('Loss per batch')
#     plt.xlabel('Batch')
#     plt.ylabel('Loss')
#     plt.show()  


# def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
#     """
#     Uses Matplotlib to visualize the correct and incorrect results of our model.
#     :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
#     :param probabilities: the output of model.call(), shape (50, num_classes)
#     :param image_labels: the labels from get_data(), shape (50, num_classes)
#     :param first_label: the name of the first class, "cat"
#     :param second_label: the name of the second class, "dog"

#     NOTE: DO NOT EDIT

#     :return: doesn't return anything, two plots should pop-up, one for correct results,
#     one for incorrect results
#     """
#     # Helper function to plot images into 10 columns
#     def plotter(image_indices, label): 
#         nc = 10
#         nr = math.ceil(len(image_indices) / 10)
#         fig = plt.figure()
#         fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
#         for i in range(len(image_indices)):
#             ind = image_indices[i]
#             ax = fig.add_subplot(nr, nc, i+1)
#             ax.imshow(image_inputs[ind], cmap="Greys")
#             pl = first_label if predicted_labels[ind] == 0.0 else second_label
#             al = first_label if np.argmax(
#                 image_labels[ind], axis=0) == 0 else second_label
#             ax.set(title="PL: {}\nAL: {}".format(pl, al))
#             plt.setp(ax.get_xticklabels(), visible=False)
#             plt.setp(ax.get_yticklabels(), visible=False)
#             ax.tick_params(axis='both', which='both', length=0)
        
#     predicted_labels = np.argmax(probabilities, axis=1)
#     num_images = image_inputs.shape[0]

#     # Separate correct and incorrect images
#     correct = []
#     incorrect = []
#     for i in range(num_images): 
#         if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
#             correct.append(i)
#         else: 
#             incorrect.append(i)

#     plotter(correct, 'Correct')
#     plotter(incorrect, 'Incorrect')
#     plt.show()


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
 
    trainX, trainY, testX, testY, X, min_max = get_data(window_size, test_samples, future_time_steps)
    # model = Model()

    # for epoch in range(model.epochs):
    #     train(model, train_inputs, train_labels)
    # # visualize_loss(model.loss_list)
    # accuracy = test(model,test_inputs,test_labels)
    # print(accuracy)



if __name__ == '__main__':
    main()
