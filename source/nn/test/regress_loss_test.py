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
from nn.function.active_function import LinnerFunction
from nn.function.loss_function import MeanSquaredError
from nn.function.loss_function import MeanAbsoluteError
from nn.function.active_function import SoftMaxFunction
from nn.mp import Layer
import nn.test.execute_test as et
np.random.seed(1)

data_train = pd.read_csv('./../data/regression/data.cube.train.1000.csv')
data_test = pd.read_csv('./../data/regression/data.cube.train.1000.csv')
(x_train, y_train, x_test, y_test) = pre.prepare_data_train_test(data_train, data_test,normalize_x=True, normalize_y=True)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# hyper parameters
max_loops = 100
batch_size = 10000
alpha = 0.1
# end hyper parameters

# statistics
mpl_costs = []
mpl_precisions = []
num_per_case = 3

mpls = []
# relu
layers = []
layers.append(Layer(1, None))
layers.append(Layer(1, LinnerFunction()))

figure, axis = plt.subplots(2, num_per_case)
figure_bounary, axis_bounary = plt.subplots(2, num_per_case)
#MeanSquaredError
for i in range(0,num_per_case):

    mpl = MultilayerPerceptron(layers,MeanSquaredError(), weight_scale=0.5)
    (cost, train_deviation_error, train_mean_error,train_deviation_error,train_mean_error)=et.exec_regression(mpl,x_train,y_train,x_test, y_test,max_loops,batch_size, alpha)
    mpl_costs.append(cost)
    mpl_precisions.append(("MSE-" + str(i), train_deviation_error, train_mean_error,train_deviation_error,train_mean_error))
    plot.plot_cost(cost,axis, 0,i  ,"MSE-" + str(i))
    mpls.append(mpl)
    plot.plot_regression_curl_axis(x_train[:,0:100],y_train[:,0:100],x_test[:,0:100],y_test[:,0:100],mpl,axis_bounary, 0,i,"MSE-"+ str(i))

# MeanAbsoluteError
for i in range(0,num_per_case):

    mpl = MultilayerPerceptron(layers,MeanAbsoluteError(),weight_scale=0.1)
    (cost, train_deviation_error, train_mean_error,train_deviation_error,train_mean_error)=et.exec_regression(mpl,x_train,y_train,x_test, y_test,max_loops,batch_size, alpha)
    mpl_costs.append(cost)
    mpl_precisions.append(("MAE-" + str(i), train_deviation_error, train_mean_error,train_deviation_error,train_mean_error))
    plot.plot_cost(cost,axis, 1,i,"MAE-" + str(i))
    mpls.append(mpl)
    # if i == 0:
    #     plot.plot_multipl_decision_region(x_train[:,0:100],y_train[:,0:100],3,mpl)
    plot.plot_regression_curl_axis(x_train[:,0:100],y_train[:,0:100],x_test[:,0:100],y_test[:,0:100],mpl,axis_bounary, 1,i,"MAE-"+ str(i))
#

plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.show()

for x, y1, y2, z1, z2 in mpl_precisions:
    print(x, y1, y2, z1, z2 )