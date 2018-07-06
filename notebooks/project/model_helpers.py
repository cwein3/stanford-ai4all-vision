import numpy as np
from scipy.special import expit as s_curve
from utils.logistic_regression_utils import *
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def predict_probability(data, weights):
    """
    Use the data and weights to calculate a probability for each data point.
    For example, if "data" has 100 rows, this function should return an array
    with 100 numbers between 0 and 1.

    HINT: "dot products" can be done with np.dot(...)
    HINT: Use the sigmoid function which can be called with s_curve(...)
    HINT: data is of shape (dataset size, num features), and weights is 
    of shape (num features, 1)
    """
    pred = None
    ## YOUR CODE HERE
    ## END YOUR CODE
    return pred[...,None]

def sgd(data, labels, weights, learning_rate, regularization_rate):
    """
    Loop over all the data and labels, one at a time, and update the weights using the logistic
    regression learning rule.

    HINT: We already call on "predict_probability" for you for this function. Try to use the output 
    in the logistic regression learning rule!
    HINT: As before, data is of shape (dataset size, num features), and weights is 
    of shape (num features, 1)
    HINT: during each iteration of the loop, you call predict probability, apply the logistic
    regression rule, and then perform the regularization update.
    """
    for i in range(data.shape[0]):
        prob = predict_probability(data[i, :], weights)
        ## YOUR CODE HERE
        ## END YOUR CODE

    return weights

def batch_sgd(data, labels, weights, learning_rate, regularization_rate, batch_size):
    """
    Loop over all the data and labels and update the weights using the logistic
    regression learning rule, averaged over multiple samples.

    HINT: You should use the "create_batches" function below.
    HINT: This function will be very similar to "sgd", but you will need to use
    np.mean(...) to average up multiple gradients.
    """
    data_batch, labels_batch = create_batches(data, labels, batch_size)
    
    for ind, curr_batch in enumerate(data_batch):
        label_batch = labels_batch[ind] # stores the labels for the current batch
        ## YOUR CODE HERE
        ## END YOUR CODE

    return weights

def create_batches(data, labels, batch_size):
    data_batch = np.array_split(data, len(data)/batch_size)
    labels_batch = np.array_split(labels, len(labels)/batch_size)
    return data_batch, labels_batch

