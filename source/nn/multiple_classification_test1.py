# read from csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nn.function.util.plot_util as plot
import nn.function.util.one_hot as oh
import matplotlib.image as mping
import math
from nn.mp import MultilayerPerceptron
from nn.function.active_function import ReluFunction
from nn.function.active_function import SoftMaxFunction
from nn.function.loss_function import CrossEntropy
from nn.mp import Layer

data = pd.read_csv('./data/classification/data.three_gauss.train.1000.csv')
#print(data.head(5))
s = data.values.shape
np.random.seed(1234)
data = data.values
np.random.shuffle(data)
data = data.T
x_train = data[0:2,  :]
y_labels = data[-1,  :]
y_labels = y_labels.reshape(1,y_labels.shape[0])
unique_labels = np.unique(y_labels)
unique_labels = unique_labels.reshape((unique_labels.shape[0],1))
y_train = np.zeros((len(unique_labels), y_labels.shape[1]))
y_train = oh.to_one_hot(y_labels)
print(x_train[:, 1:10])
print(y_train[:, 1:10])
print(y_train.shape)


# build network
layers = []
relu = ReluFunction()
# layers.append(Layer(2, None))
# layers.append(Layer(4, ReluFunction()))
# layers.append(Layer(3, SoftMaxFunction()))

layers.append(Layer(2, None))
layers.append(Layer(4, ReluFunction()))
#layers.append(Layer(7, ReluFunction()))
layers.append(Layer(5, ReluFunction()))
layers.append(Layer(3, SoftMaxFunction()))

multilayerPerceptron = MultilayerPerceptron(x_train, y_train, layers, CrossEntropy())
costs = multilayerPerceptron.train(6000, 3000 , 0.05 ,0.3)
print("costs=")
print(costs)
plt.plot(range(len(costs)), costs)
plt.xlabel('Grident steps')
plt.xlabel('costs')
plt.show()

predict_y = multilayerPerceptron.test(x_train)
predict_y = np.where(predict_y > 0.5, 1, 0)
plt.xlabel('==========')

train_precision = (predict_y*y_train).sum() / y_train.shape[1] * 100
print("train_precision=" + str(train_precision))

data_test = pd.read_csv('./data/classification/data.three_gauss.test.1000.csv')
data_test=data_test.values
#print("``````weitht=" + str(multilayerPerceptron.weights))
np.random.shuffle(data_test)
data_test = data_test.T;
x_test= data_test[0:2, :]
y_test = data_test[-1, :]
y_test = oh.to_one_hot(y_test)


predict_test_y = multilayerPerceptron.test(x_test)
predict_test_y = np.where(predict_test_y > 0.5, 1, 0)
print("predict y=" + str(predict_test_y[:, 0:10]))

test_precision = (predict_test_y*y_test).sum()/ y_test.shape[1] * 100
print("test_precision=" + str(test_precision))

plot.plot_multipl_decision_region(x_train[:,0:100],y_train[:,0:100],3,multilayerPerceptron)
plot.plot_multipl_decision_region(x_test[:,0:100],y_test[:,0:100],3,multilayerPerceptron)