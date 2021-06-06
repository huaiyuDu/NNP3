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
np.random.seed(1)
data = pd.read_csv('./data/train.csv')

train_data = data.sample(frac = 0.8)
validation_data = data.drop(train_data.index)

train_data = train_data.values
validation_data = validation_data.values

num_training_examples = 5000

x_train = train_data[:num_training_examples,1:]
(x_train, mean, deviation) = norm.normalize(x_train)
x_train = x_train.T

y_train = train_data[:num_training_examples,[0]]
y_train = oh.to_one_hot(y_train).T

x_validation = validation_data[:,1:]
(x_validation, mean, deviation) = norm.normalize(x_validation,mean, deviation)
x_validation = x_validation.T

y_validation = validation_data[:,[0]]
y_validation = oh.to_one_hot(y_validation).T


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
#layers.append(Layer(25, ReluFunction()))
layers.append(Layer(10, SoftMaxFunction()))

multilayerPerceptron = MultilayerPerceptron(layers,BinaryCrossEntropy(),weight_scale=0.05)
costs = multilayerPerceptron.train(x_train, y_train, 2000, 5000 , 0.1 )
plt.plot(range(len(costs)), costs)
plt.xlabel('Grident steps')
plt.xlabel('costs')
plt.show()

predict_y = multilayerPerceptron.test(x_train)
predict_y = np.where(predict_y > 0.5, 1, 0)
plt.xlabel('==========')

train_precision = (predict_y*y_train).sum() / y_train.shape[1] * 100
print("train_precision=" + str(train_precision))




predict_validation_y = multilayerPerceptron.test(x_validation)
predict_validation_y = np.where(predict_validation_y > 0.5, 1, 0)
#print("predict y=" + str(predict_test_y[:, 0:10]))

validation_precision = (predict_validation_y*y_validation).sum()/ y_validation.shape[1] * 100
print("validation_precision=" + str(validation_precision))

#plot.plot_multipl_decision_region(x_train[:,0:100],y_train[:,0:100],10,multilayerPerceptron)
#plot.plot_multipl_decision_region(x_test[:,0:100],y_test[:,0:100],10,multilayerPerceptron)