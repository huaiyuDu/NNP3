# read from csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nn.function.util.plot_util as plot

import matplotlib.image as mping
import math
from nn.mp import MultilayerPerceptron
from nn.function.active_function import ReluFunction
from nn.function.active_function import TanhFunction
from nn.function.active_function import SigmoidFunction
from nn.function.loss_function import BinaryCrossEntropy
from nn.mp import Layer

np.random.seed(1)
data = pd.read_csv('./data/classification/data.simple.train.1000.csv')
print(data.head(5))
s = data.values.shape
data= data.values
np.random.shuffle(data)
data = data.T;
x_train = data[0:2, :]
y_train = data[-1, :]
y_train = y_train - 1
y_train = y_train.reshape(1,y_train.shape[0])
print(x_train)
print(y_train)
print(y_train.shape)


# build network

layers = []
layers.append(Layer(2, None))
layers.append(Layer(3, TanhFunction()))
layers.append(Layer(3, TanhFunction()))
layers.append(Layer(1, SigmoidFunction()))
multilayerPerceptron = MultilayerPerceptron(layers,BinaryCrossEntropy(),weight_scale=0.3)
costs = multilayerPerceptron.train(x_train, y_train, 3000, 10000 , 0.1 )
print("costs=")
print(costs)
plt.plot(range(len(costs)), costs)
plt.xlabel('Grident steps')
plt.xlabel('costs')
plt.show()

predict_y = multilayerPerceptron.test(x_train)
predict_y = np.where(predict_y > 0.5, 1, 0)
train_precision = np.sum(predict_y == y_train) / y_train.shape[1] * 100
print("train_precision=" + str(train_precision))

data_test = pd.read_csv('./data/classification/data.simple.test.1000.csv')

data_test = data_test.values.T;
x_test= data_test[0:2, :]
y_test = data_test[-1, :]
y_test = y_test - 1
y_test = y_test.reshape(1,y_test.shape[0])

predict_test_y = multilayerPerceptron.test(x_test)
predict_test_y = np.where(predict_test_y > 0.5, 1, 0)
test_precision = np.sum(predict_test_y == y_test) / y_test.shape[1] * 100
print("test_precision=" + str(test_precision))

plot.plot_decision_region(x_test[:,0:100],y_test[:,0:100],multilayerPerceptron)