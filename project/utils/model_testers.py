from project.model_helpers import *
import numpy as np

def test_predict_probability():
    true_pred = np.array([[0.009727],
                         [ 0.80695073],
                         [ 0.47222148],
                         [ 0.50962809],
                         [ 0.94832398],
                         [ 0.39283667],
                         [ 0.94318405],
                         [ 0.54676926],
                         [ 0.26353454],
                         [ 0.80562677]])
    np.random.seed(0)
    data = np.random.randn(10,5)
    weights = np.random.randn(5)
    pred = predict_probability(data, weights)
    if np.allclose(true_pred, pred):
        print('Your "predict_probability" function seems to be working!')
    else:
        print('It looks like there\'s a bug in your "predict_probability" function')

def test_sgd():
    true_weights = np.array(
        [ 1.09523196, -0.43548978,  0.41456338, -0.04298213,  1.44849388]
        )

    np.random.seed(0)
    data = np.random.randn(10,5)
    labels = np.random.randint(0,2,(10,))
    weights = np.random.randn(5)
    learning_rate = 0.01
    regularization_rate = 0.001

    weights = sgd(data, labels, weights, learning_rate, regularization_rate)
    if np.allclose(true_weights, weights):
        print('Your "sgd" function seems to be working!')
    else:
        print('It looks like there\'s a bug in your "sgd" function')

def test_batch_sgd():
    true_weights = np.array(
        [ 0.6217815, 1.78235447, 0.16797774, -1.71221836, 0.16394118, -0.8614746 ]
        )

    np.random.seed(0)
    data = np.random.randn(10,6)
    labels = np.random.randint(0,2,(10,)).reshape((10, 1))
    weights = np.random.randn(6)
    learning_rate = 0.01
    regularization_rate = 0.001
    batch_size = 5

    weights = batch_sgd(data, labels, weights, learning_rate, regularization_rate, batch_size)
    if np.allclose(true_weights, weights):
        print('Your "batch_sgd" function seems to be working!')
    else:
        print('It looks like there\'s a bug in your "batch_sgd" function')

if __name__ == "__main__":
    test_predict_probability()
    test_sgd()
    test_batch_sgd()
