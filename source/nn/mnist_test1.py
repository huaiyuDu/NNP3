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
from nn.function.loss_function import BinaryCrossEntropy
from nn.mp import Layer
from nn.function.util import normalize as norm

data = pd.read_csv('./data/mnist-demo.csv')

train_data = data.sample(frac = 0.8)
test_data = data.drop(train_data.index)

train_data = train_data.values
test_data = test_data.values

num_training_examples = 5000

x_train = train_data[:num_training_examples,1:]
(x_train, mean, deviation) = norm.normalize(x_train)
x_train = x_train.T

y_train = train_data[:num_training_examples,[0]]
y_train = oh.to_one_hot(y_train.T)

x_test = test_data[:,1:]
(x_test, mean, deviation) = norm.normalize(x_test,mean, deviation)
x_test = x_test.T

y_test = test_data[:,[0]]
y_test = oh.to_one_hot(y_test.T)


print(x_train[:, 1:10])
print(y_train[:, 1:10])
print(y_train.shape)


# build network
layers = []
relu = ReluFunction()
# layers.append(Layer(2, None))
# layers.append(Layer(4, ReluFunction()))
# layers.append(Layer(3, SoftMaxFunction()))

layers.append(Layer(784, None))
layers.append(Layer(25, ReluFunction()))
layers.append(Layer(10, SoftMaxFunction()))

multilayerPerceptron = MultilayerPerceptron(x_train, y_train, layers,BinaryCrossEntropy())
costs = multilayerPerceptron.train(1000, 5000 , 0.1 )
plt.plot(range(len(costs)), costs)
plt.xlabel('Grident steps')
plt.xlabel('costs')
plt.show()

predict_y = multilayerPerceptron.test(x_train)
predict_y = np.where(predict_y > 0.5, 1, 0)
plt.xlabel('==========')

train_precision = (predict_y*y_train).sum() / y_train.shape[1] * 100
print("train_precision=" + str(train_precision))




predict_test_y = multilayerPerceptron.test(x_test)
predict_test_y = np.where(predict_test_y > 0.5, 1, 0)
#print("predict y=" + str(predict_test_y[:, 0:10]))

test_precision = (predict_test_y*y_test).sum()/ y_test.shape[1] * 100
print("test_precision=" + str(test_precision))

#plot.plot_multipl_decision_region(x_train[:,0:100],y_train[:,0:100],10,multilayerPerceptron)
#plot.plot_multipl_decision_region(x_test[:,0:100],y_test[:,0:100],10,multilayerPerceptron)