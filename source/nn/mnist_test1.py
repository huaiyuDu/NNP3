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


max_num=5000
x_train = train_data[:,1:]
(x_train, mean, deviation) = norm.normalize(x_train)
x_train = x_train.T

y_train = train_data[:,[0]]
y_train = oh.one_hot_encoding(y_train, 10).T

x_validation = validation_data[:,1:]
(x_validation, mean, deviation) = norm.normalize(x_validation,mean, deviation)
x_validation = x_validation.T

y_validation = validation_data[:,[0]]
y_validation = oh.one_hot_encoding(y_validation, 10).T


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
costs,costs_validate,costs_train,weight_his,bias_his = multilayerPerceptron.train(x_train, y_train, 3000, 50000 , 0.1 ,
                                                                                  validateX = x_validation,validateY=y_validation)
plt.plot(range(len(costs)), costs)

plt.xlabel('Grident steps')
plt.ylabel('costs')
plt.show()
plt.plot(range(len(costs_validate)), costs_validate)
plt.plot(range(len(costs_train)), costs_train)
plt.xlabel('Grident steps')
plt.ylabel('costs')
plt.show()
# reset weight
index = np.argmin(costs_validate)
print(np.argmin(costs_validate))
multilayerPerceptron.weights = weight_his[index]
multilayerPerceptron.biases = bias_his[index]
predict_y = multilayerPerceptron.test(x_train)
predict_y = np.where(predict_y > 0.5, 1, 0)
plt.xlabel('==========')
print('mini validation cost:')


train_precision = (predict_y*y_train).sum() / y_train.shape[1] * 100
print("train_precision=" + str(train_precision))

data_test = pd.read_csv('./data/test.csv')
data_test= data_test.values
x_test = data_test
(x_test, mean, deviation) = norm.normalize(x_test)
x_test = x_test.T

#y_test = data_test[:,[0]]
#y_test = oh.one_hot_encoding(y_test, 10).T

predict_test_y = multilayerPerceptron.test(x_test)
predict_test_y = np.where(predict_test_y > 0.5, 1, 0)
#test_precision = (predict_test_y*y_test).sum()/ y_test.shape[1] * 100
print(predict_test_y)
test_y_res = np.argmax(predict_test_y,axis=0)
print(test_y_res)
#print("test_precision=" + str(test_precision))
resultfile = open("result.txt", "w")
resultfile.write("ImageId,Label\n")
for i in range(len(test_y_res)):
    resultfile.write(str(i+1) +"," +str(test_y_res[i])+"\n")
resultfile.close()
predict_validation_y = multilayerPerceptron.test(x_validation)
predict_validation_y = np.where(predict_validation_y > 0.5, 1, 0)
#print("predict y=" + str(predict_test_y[:, 0:10]))

validation_precision = (predict_validation_y*y_validation).sum()/ y_validation.shape[1] * 100
print("validation_precision=" + str(validation_precision))

#plot.plot_multipl_decision_region(x_train[:,0:100],y_train[:,0:100],10,multilayerPerceptron)
#plot.plot_multipl_decision_region(x_test[:,0:100],y_test[:,0:100],10,multilayerPerceptron)