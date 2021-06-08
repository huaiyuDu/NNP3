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
data_train = pd.read_csv('./data/classification/data.three_gauss.train.1000.csv')
data_test = pd.read_csv('./data/classification/data.three_gauss.test.1000.csv')
(x_train, y_train, x_test, y_test) = pre.prepare_data_train_test(data_train, data_test,one_hot_encoding=True)


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

plot.plot_multipl_decision_region(x_train[:,0:100],y_train[:,0:100],3,multilayerPerceptron)
plot.plot_multipl_decision_region(x_test[:,0:100],y_test[:,0:100],3,multilayerPerceptron)

unrolled_weight_his = plot.unroll_weight_his(multilayerPerceptron.weight_his)
plt.plot(unrolled_weight_his[:,0])
#plt.plot(unrolled_weight_his[:,1])
plt.xlabel = "weight history"
plt.show()
unrolled_delta_weight_his = plot.unroll_weight_his(multilayerPerceptron.delta_weight_his)
plt.plot(unrolled_delta_weight_his[:,0])
#plt.plot(unrolled_delta_weight_his[:,1])
plt.xlabel= "delta weight history"
plt.show()