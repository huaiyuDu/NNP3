# read from csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nn.function.util.plot_util as plot
import nn.function.util.one_hot as oh
import nn.function.util.prepare_data as pre
import matplotlib.image as mping
import math
from nn.mp import MultilayerPerceptron
from nn.function.active_function import ReluFunction
from nn.function.active_function import SoftMaxFunction
from nn.function.loss_function import CrossEntropy
from nn.mp import Layer
np.random.seed(1)
data_train = pd.read_csv('./data/train.csv')
data_test = pd.read_csv('./data/test.csv')
label_column = data_train['label']
data_train.drop(labels=['label'], axis=1, inplace = True)
data_train.insert(1, 'label', label_column)
label_column = data_test['label']
data_test.drop(labels=['label'], axis=1, inplace = True)
data_test.insert(1, 'label', label_column)

(x_train, y_train, x_test, y_test) = pre.prepare_data_train_test(data_train, data_test,one_hot_encoding=True)


# build network
layers = []
layers.append(Layer(784, None))
layers.append(Layer(25, ReluFunction()))
layers.append(Layer(10, SoftMaxFunction()))

multilayerPerceptron = MultilayerPerceptron(layers, CrossEntropy(),weight_scale=0.3)
costs = multilayerPerceptron.train(x_train, y_train, 6000, 3000 , 0.1 ,0.3)
print("costs=")
#print(costs)
plt.plot(range(len(costs)), costs)
plt.xlabel('Grident steps')
plt.xlabel('costs')
plt.show()

predict_y = multilayerPerceptron.test(x_train)
predict_y = np.where(predict_y > 0.5, 1, 0)

train_precision = (predict_y*y_train).sum() / y_train.shape[1] * 100
print("train_precision=" + str(train_precision))


predict_test_y = multilayerPerceptron.test(x_test)
predict_test_y = np.where(predict_test_y > 0.5, 1, 0)
print("predict y=" + str(predict_test_y[:, 0:10]))

test_precision = (predict_test_y*y_test).sum()/ y_test.shape[1] * 100
print("test_precision=" + str(test_precision))
