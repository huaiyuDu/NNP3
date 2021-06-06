# read from csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nn.function.util.plot_util as plot
import matplotlib.image as mping
import math
from nn.mp import MultilayerPerceptron
from nn.function.active_function import ReluFunction
from nn.function.active_function import SigmoidFunction
from nn.function.active_function import LeakyReluFunction
from nn.function.active_function import LinnerFunction
from nn.function.util import normalize as norm
from nn.function.loss_function import MeanSquaredError
from nn.function.util import print_util as pu
from nn.mp import Layer

#data = pd.read_csv('./data/regression/data.activation.train.1000.csv')
data = pd.read_csv('./data/regression/data.cube.train.1000.csv')
print(data.head(5))
s = data.values.shape
data = data.values
(data, min, range_width) = norm.normalize_zero_to_one(data)
np.random.shuffle(data)
data = data.T;
x_train = data[0, :]
y_train = data[-1, :]
x_train= x_train.reshape(1,x_train.shape[0])
y_train = y_train.reshape(1,y_train.shape[0])
print(x_train)
print(y_train)
print(y_train.shape)


# build network
layers = []
relu = ReluFunction()
# layers.append(Layer(1, None))
# layers.append(Layer(4, SigmoidFunction()))
# #layers.append(Layer(6, ReluFunction()))
# layers.append(Layer(4, SigmoidFunction()))
# layers.append(Layer(1, LinnerFunction()))
layers.append(Layer(1, None))
layers.append(Layer(4, SigmoidFunction()))
layers.append(Layer(4, SigmoidFunction()))
#layers.append(Layer(6, SigmoidFunction()))
#layers.append(Layer(6, ReluFunction()))
#layers.append(Layer(6, ReluFunction()))
layers.append(Layer(1, SigmoidFunction()))

multilayerPerceptron = MultilayerPerceptron(layers,MeanSquaredError())
costs = multilayerPerceptron.train(x_train, y_train, 60000, 1000, 0.2, 0.1)
print("costs=")
print(costs)
plt.plot(range(len(costs)), costs)
plt.xlabel('Grident steps')
plt.xlabel('costs')
plt.show()

predict_y = multilayerPerceptron.test(x_train)
error_test = predict_y- y_train
deviation_error = np.std(error_test)
mean_error = np.mean(error_test)
print("deviation_error=" + str(deviation_error))
print("mean_error=" + str(mean_error))
#plot.plot_regression_curl(x_train,y_train,multilayerPerceptron)
# predict_y = np.where(predict_y > 0.5, 1, 0)
# train_precision = np.sum(predict_y == y_train) / y_train.shape[1] * 100
# print("train_precision=" + str(train_precision))
#
#data_test = pd.read_csv('./data/regression/data.activation.test.1000.csv')
data_test = pd.read_csv('./data/regression/data.cube.test.1000.csv')
data_test = data_test.values
(data_test,min,  deviation) = norm.normalize_zero_to_one(data_test,min,range_width)
np.random.shuffle(data_test)
data_test = data_test.T;
x_test = data_test[0, :]
y_test = data_test[-1, :]
x_test = x_test.reshape(1,x_test.shape[0])
y_test = y_test.reshape(1,y_test.shape[0])
#
predict_test_y = multilayerPerceptron.test(x_test)
error_test = predict_test_y- y_test
test_deviation_error = np.std(error_test)
test_mean_error = np.mean(error_test)
print("test_deviation_error=" + str(test_deviation_error))
print("test_mean_error=" + str(test_mean_error))
plot.plot_regression_curl(x_train[:,0:100],y_train[:,0:100],x_test[:,0:100],y_test[:,0:100],multilayerPerceptron)
pu.print_weights_biases(multilayerPerceptron.weights,multilayerPerceptron.biases)

