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
from nn.function.loss_function import CrossEntropy
from nn.function.active_function import SoftMaxFunction
from nn.mp import Layer
import nn.test.execute_test as et
np.random.seed(1)
data_train = pd.read_csv('./../data/classification/data.circles.train.10000.csv')
data_test = pd.read_csv('./../data/classification/data.circles.test.10000.csv')
(x_train, y_train, x_test, y_test) = pre.prepare_data_train_test(data_train, data_test,one_hot_encoding=True)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# hyper parameters
max_loops = 15000
batch_size = 10000
alpha = 0.1
# end hyper parameters

# statistics
mpl_costs = []
mpl_precisions = []
num_per_case = 1

mpls = []
# relu
layers = []
layers.append(Layer(2, None))
layers.append(Layer(4, ReluFunction()))
layers.append(Layer(4, SoftMaxFunction()))

figure, axis = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.5,hspace=0.5)
figure_bounary, axis_bounary = plt.subplots(2, 2)
for i in range(0,num_per_case):

    mpl = MultilayerPerceptron(layers,CrossEntropy(), weight_scale=0.5)
    (cost, train_precision, test_precision)=et.exec_multipl_cls(mpl,x_train,y_train,x_test, y_test,max_loops,batch_size, alpha)
    mpl_costs.append(cost)
    mpl_precisions.append(("4 neurons-" + str(i), train_precision,test_precision))
    plot.plot_cost(cost,axis, 0,0 ,"4 neurons-" + str(i))
    mpls.append(mpl)
    # if i == 0:
    #     plot.plot_multipl_decision_region(x_train[:,0:100],y_train[:,0:100],3,mpl)
    plot.plot_multipl_decision_region_axis(x_train[:,0:1000],y_train[:,0:1000],4,mpl,axis_bounary, 0,0,"4 neurons-"+ str(i))

# sigmoid
layers = []
layers.append(Layer(2, None))
layers.append(Layer(8, ReluFunction()))
layers.append(Layer(4, SoftMaxFunction()))
for i in range(0,num_per_case):

    mpl = MultilayerPerceptron(layers,CrossEntropy(), weight_scale=0.5)
    (cost, train_precision, test_precision)=et.exec_multipl_cls(mpl,x_train,y_train,x_test, y_test,max_loops,batch_size, alpha)
    mpl_costs.append(cost)
    mpl_precisions.append(("8 neurons-" + str(i), train_precision,test_precision))
    plot.plot_cost(cost,axis, 0,1 ,"8 neurons-" + str(i))
    mpls.append(mpl)
    # if i == 0:
    #     plot.plot_multipl_decision_region(x_train[:,0:100],y_train[:,0:100],3,mpl)
    plot.plot_multipl_decision_region_axis(x_train[:,0:1000],y_train[:,0:1000],4,mpl,axis_bounary, 0,1,"8 neurons-"+ str(i))

# # tanh
layers = []
layers.append(Layer(2, None))
layers.append(Layer(12, ReluFunction()))
layers.append(Layer(4, SoftMaxFunction()))
for i in range(0,num_per_case):

    mpl = MultilayerPerceptron(layers,CrossEntropy(), weight_scale=0.5)
    (cost, train_precision, test_precision)=et.exec_multipl_cls(mpl,x_train,y_train,x_test, y_test,max_loops,batch_size, alpha)
    mpl_costs.append(cost)
    mpl_precisions.append(("12 neurons-" + str(i), train_precision,test_precision))
    plot.plot_cost(cost,axis, 1,0 ,"12 neurons-" + str(i))
    mpls.append(mpl)
    # if i == 0:
    #     plot.plot_multipl_decision_region(x_train[:,0:100],y_train[:,0:100],3,mpl)
    plot.plot_multipl_decision_region_axis(x_train[:,0:1000],y_train[:,0:1000],4,mpl,axis_bounary, 1,0,"12 neurons-"+ str(i))

layers = []
layers.append(Layer(2, None))
layers.append(Layer(16, ReluFunction()))
layers.append(Layer(4, SoftMaxFunction()))
for i in range(0,num_per_case):

    mpl = MultilayerPerceptron(layers,CrossEntropy(), weight_scale=0.5)
    (cost, train_precision, test_precision)=et.exec_multipl_cls(mpl,x_train,y_train,x_test, y_test,max_loops,batch_size, alpha)
    mpl_costs.append(cost)
    mpl_precisions.append(("16 neurons-" + str(i), train_precision,test_precision))
    plot.plot_cost(cost,axis, 1,1 ,"16 neurons-" + str(i))
    mpls.append(mpl)
    # if i == 0:
    #     plot.plot_multipl_decision_region(x_train[:,0:100],y_train[:,0:100],3,mpl)
    plot.plot_multipl_decision_region_axis(x_train[:,0:1000],y_train[:,0:1000],4,mpl,axis_bounary, 1,1,"16 neurons-"+ str(i))


plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.show()
for i in range (0,4*num_per_case,num_per_case):
    plot.plot_multipl_decision_region(x_train[:,0:1000],y_train[:,0:1000],4,mpls[i])
for x, y, z in mpl_precisions:
    print(x, y, z)