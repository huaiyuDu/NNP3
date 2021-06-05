import numpy as np


def exec_binary_cls(mlp, x_train, y_train, x_test, y_test, max_times, batch_size, alpha, momentum=0):
    cost = mlp.train(x_train, y_train, max_times, batch_size, alpha, momentum)
    y_train_predict = mlp.test(x_train)
    y_test_predict = mlp.test(x_test)
    y_train_predict = np.where(y_train_predict > 0.5, 1, 0)
    y_test_predict = np.where(y_test_predict > 0.5, 1, 0)
    train_precision = np.sum(y_train_predict == y_train) / y_train.shape[1] * 100
    test_precision = np.sum(y_test_predict == y_test) / y_test.shape[1] * 100
    return cost, train_precision, test_precision

def exec_multipl_cls(mlp, x_train, y_train, x_test, y_test, max_times, batch_size, alpha, momentum=0):
    cost = mlp.train(x_train, y_train, max_times, batch_size, alpha, momentum)
    y_train_predict = mlp.test(x_train)
    y_test_predict = mlp.test(x_test)
    y_train_predict = np.where(y_train_predict > 0.5, 1, 0)
    y_test_predict = np.where(y_test_predict > 0.5, 1, 0)

    train_precision = (y_train_predict*y_train).sum()/ y_train.shape[1] * 100
    test_precision = (y_test_predict*y_test).sum()/ y_test.shape[1] * 100
    return cost, train_precision, test_precision
