import numpy as np
import pandas as pd
from nn.function.util import normalize as norm
from nn.function.util import polynomial_util as poly
from nn.function.util import one_hot as onehot


def prepare_data_train_test(trainData, testData, normalize_x=False, normalize_y=False, polinomial_dgree=0,
                     one_hot_encoding=False):
    (x_train, y_train, x_mean, x_deviation, y_mean, y_deviation) = prepare_data(
        trainData, normalize_x, normalize_y, polinomial_dgree, one_hot_encoding)
    (x_test, y_test, x_mean, x_deviation, y_mean, y_deviation) = prepare_data(
        testData, normalize_x, normalize_y, polinomial_dgree, one_hot_encoding, x_mean, x_deviation, y_mean,
        y_deviation)
    return x_train, y_train, x_test, y_test


def prepare_data(data, normalize_x=False, normalize_y=False,
                            polynomial_degree=0, one_hot_encoding=False,
                 x_mean=None, x_deviation=None, y_mean=None, y_deviation=None):
    data = data.values
    np.random.shuffle(data)

    x = data[:, 0:-1]

    x = np.array(x,dtype=np.float64)
    y = data[:, -1]
    # reshape to 2D array if needed
    if len(x.shape) == 1:
        x = x.reshape(x.shape[0], 1)
    y = y.reshape(y.shape[0], 1)

    if normalize_x:
        (x, x_mean, x_deviation) = norm.normalize(x, x_mean, x_deviation)
    if normalize_y:
        (y, y_mean, y_deviation) = norm.normalize(y, y_mean, y_deviation)

    if polynomial_degree > 1:
        x = poly.expend_to_polynomial(x, polynomial_degree)

    if one_hot_encoding:
        y = onehot.to_one_hot(y)

    return x.T, y.T, x_mean, x_deviation, y_mean, y_deviation
