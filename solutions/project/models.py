import numpy as np
from sklearn.metrics import accuracy_score

from utils.logistic_regression_utils import *
from project.model_helpers import sgd, batch_sgd, predict_probability

class LogisticRegression():
    """
    A class for the logistic regression classifier. It has a "predict" method
    for assigning labels to inputs, and a "train" method for updating its
    weights.
    """
    def __init__(self, num_features, learning_rate, regularization_rate,
            batch_size, epochs):
        self._epochs = epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._regularization_rate = regularization_rate
        self._weights = np.random.randn(num_features)
        self.name = "Logistic Regression(lr:{} ; rr:{} ; bs:{})".format(
                learning_rate,
                regularization_rate,
                batch_size
                )

    def predict(self, data):
        weights = self._weights
        pred = predict_probability(data, weights)
        return pred

    def train(self, data, labels):
        weights = self._weights
        learning_rate = self._learning_rate
        regularization_rate = self._regularization_rate
        batch_size = self._batch_size
        epochs = self._epochs

        losses = []
        for e in range(epochs):
            if check_training_progress(weights, e, epochs): break
            # if the batch size is above one, use batch sgd
            if batch_size > 1:
                weights = batch_sgd(data, labels, weights, learning_rate,
                        regularization_rate, batch_size)
            # else use normal Stochastic Gradient Descent
            else:
                weights = sgd(data, labels, weights, learning_rate, regularization_rate)
            loss = self.compute_loss(data, labels)
            losses.append(loss)
        self._weights = weights
        #return losses
        visualize_epochs(self,losses)

    def compute_loss(self, data, labels):
        p = np.squeeze(self.predict(data))
        labels = np.squeeze(labels)
        loss = p - p * labels + np.log(1. + np.exp(-p))
        loss = np.mean(loss)
        return loss

    def compute_accuracy(self, data, labels):
        pred = (self.predict(data) > 0.5).astype("int32")
        acc = accuracy_score(pred, labels)
        return acc

class RandomGuesser():
    """
    A model that doesn't do any "learning". Just randomly
    guesses labels to make a prediction, like flipping a coing.
    """
    def __init__(self, *args):
        self.name = "Random Guesser"

    def predict(self, data):
        pred = np.random.randint(0,2,(len(data)))
        return pred

    def train(self, data, labels, epochs=100):
        pass

    def compute_accuracy(self, data, labels):
        pred = (self.predict(data) > 0.5).astype("int32")
        acc = accuracy_score(pred, labels)
        return acc

class MajorityGuesser():
    """
    A model that just looks at the most frequent label in the dataset, and just
    picks that over and over. For example, if there are more Trucks than Planes
    in your training set, it will just guess everything is a Truck.
    """
    def __init__(self, *args):
        self.name = "Majority Guesser"

    def predict(self, data):
        pred = np.ones(len(data)) * self._majority
        return pred

    def train(self, data, labels, epochs=100):
        self._majority = mode(labels)[0][0]

    def compute_accuracy(self, data, labels):
        pred = (self.predict(data) > 0.5).astype("int32")
        acc = accuracy_score(pred, labels)
        return acc

def visualize_epochs(model, losses):
    plt.plot(losses)
    ax = plt.gca()
    ax.set_title("{}".format(model.name))
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    plt.show()

ALL_MODELS = [LogisticRegression, RandomGuesser, MajorityGuesser]

