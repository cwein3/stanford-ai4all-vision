import numpy as np
from project.models import *
from project.metrics import *
from utils.logistic_regression_utils import *
from itertools import *

def create_trucksplanes_dataset():

    """
    Creates a dataset from the trucks and planes dataset, using color
    histograms. You must slice the data and labels into a training and validation set.
    Try using roughly 10% of your training data for the validation!

    Try experimenting with different bins and values for use_hsv!
    """

    print("### Extracting Trucks and Planes Dataset ###")
    BINS = 30
    USE_HSV = False

    # features will be of shape (dataset size, num features)
    features = extract_trucksplanes_histograms(BINS, USE_HSV)

    # labels will be the same size as the dataset
    labels = load_trucksplanes_labels()

    features, labels = shuffle_data(features, labels)

    ## YOUR CODE HERE
    num_val = features.shape[0]//10
    num_train = features.shape[0] - num_val
    data_train = features[:num_train, :]
    data_validation = features[num_train:, :]
    labels_train = labels[:num_train]
    labels_validation = labels[num_train:]
    ## END YOUR CODE

    return data_train, labels_train, data_validation, labels_validation

def create_uganda_dataset():
    """
    Creates a dataset from Uganda satellite features. You must slice the data
    and labels into a training and validation set. As before, try using roughly 
    10% of your data as the validation set.

    """

    # features will be of shape (dataset size, num features)
    features = extract_uganda_features()

    # labels will be the same size as the dataset.
    labels = load_satellite_labels()

    features, labels = shuffle_data(features, labels)

    ## YOUR CODE HERE
    num_val = features.shape[0]//10
    num_train = features.shape[0] - num_val
    data_train = features[:num_train, :]
    data_validation = features[num_train:, :]
    labels_train = labels[:num_train]
    labels_validation = labels[num_train:]
    ## END YOUR CODE

    return data_train, labels_train, data_validation, labels_validation

def cross_validation(use_satellite = False):
    """
    """
    best_hyperparameters = {
            "learning_rate": 0.0,
            "regularization_rate": 0.0,
            "batch_size": 1,
            "epochs":0
            }
    best_score_so_far = 0.0

    if use_satellite:
        data_train, labels_train, data_validation, labels_validation = create_uganda_dataset()
    else:
        data_train, labels_train, data_validation, labels_validation = create_trucksplanes_dataset()

    num_features = data_train.shape[1]

    ## YOUR CODE HERE
    # Fill the following lists with the hyperparameters you intend to try.
    learning_rates = [0.05]
    regularizations = [0.001, 0.005]
    batch_sizes = [5, 10]
    epochs = [10]
    ## END YOUR CODE

    print("### STARTING CROSS VALIDATION ###")
    for params in product(learning_rates, regularizations, batch_sizes, epochs):
        ## YOUR CODE HERE
        # Hint: create a LogisticRegression object with the given hyperparameters.
        # Train and store the best score so far and best hyperparameters.
        # Compute the accuracy score using compute_accuracy in the LogisticRegression class!
        lr, reg, batch, epoch = params
        logreg_trainer = LogisticRegression(
            num_features, 
            lr, 
            reg,
            batch,
            epoch)
        logreg_trainer.train(data_train, labels_train)
        score = logreg_trainer.compute_accuracy(data_validation, labels_validation)
        if score > best_score_so_far:
            best_score_so_far = score
            best_hyperparameters = {
                "learning_rate": lr,
                "regularization_rate": reg,
                "batch_size": batch,
                "epochs": epoch
            }
        ## END YOUR CODE
    print("### FINISHED CROSS VALIDATION ###")
    return best_hyperparameters

