import pandas as pd
import nn.function.util.prepare_data as pre
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nn.function.util.plot_util as plot
import matplotlib.image as mping
import math
from nn.mp import MultilayerPerceptron
from nn.function.active_function import ReluFunction
from nn.function.active_function import SigmoidFunction
from nn.function.active_function import TanhFunction
from nn.function.loss_function import BinaryCrossEntropy
from nn.mp import Layer
import nn.test.execute_test as et
np.random.seed(1)
data_train = pd.read_csv('./../data/classification/data.simple.train.1000.csv')
data_test = pd.read_csv('./../data/classification/data.simple.test.1000.csv')
(x_train, y_train, x_test, y_test) = pre.prepare_data_train_test(data_train, data_test)
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
y_train = y_train -1
y_test = y_test -1
# hyper parameters
max_loops = 500
batch_size = 10000
alpha = 0.1
# end hyper parameters

# statistics
mpl_costs = []
mpl_precisions = []
num_per_case = 4
# relu
layers = []
layers.append(Layer(2, None))
layers.append(Layer(3, ReluFunction()))
#layers.append(Layer(3, ReluFunction()))
layers.append(Layer(1, SigmoidFunction()))

figure, axis = plt.subplots(3, num_per_case)
for i in range(0,num_per_case):

    mpl = MultilayerPerceptron(layers,BinaryCrossEntropy(),weight_scale=0.3)
    (cost, train_precision, test_precision)=et.exec_binary_cls(mpl,x_train,y_train,x_test, y_test,max_loops,batch_size, alpha)
    mpl_costs.append(cost)
    mpl_precisions.append(("relu-" + str(i), train_precision,test_precision))
    plot.plot_cost(cost,axis, 0,i  ,"relu-" + str(i))

# sigmoid
layers = []
layers.append(Layer(2, None))
layers.append(Layer(3, SigmoidFunction()))
#layers.append(Layer(3, SigmoidFunction()))
layers.append(Layer(3, SigmoidFunction()))
for i in range(0,num_per_case):

    mpl = MultilayerPerceptron(layers,BinaryCrossEntropy(),weight_scale=0.3)
    (cost, train_precision, test_precision)=et.exec_binary_cls(mpl,x_train,y_train,x_test, y_test,max_loops,batch_size, alpha)
    mpl_costs.append(cost)
    mpl_precisions.append(("sigmod-" + str(i), train_precision,test_precision))
    plot.plot_cost(cost,axis, 1,i,"sigmod-" + str(i))

# tanh
layers = []
layers.append(Layer(2, None))
layers.append(Layer(3, TanhFunction()))
#layers.append(Layer(3, TanhFunction()))
layers.append(Layer(1, SigmoidFunction()))
for i in range(0,num_per_case):

    mpl = MultilayerPerceptron(layers,BinaryCrossEntropy(),weight_scale=0.3)
    (cost, train_precision, test_precision)=et.exec_binary_cls(mpl,x_train,y_train,x_test, y_test,max_loops,batch_size, alpha)
    mpl_costs.append(cost)
    mpl_precisions.append(("tanh-" + str(i), train_precision,test_precision))
    plot.plot_cost(cost,axis, 2,i,"tanh-" + str(i))
    #print(cost)


plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.show()
print(mpl_precisions)