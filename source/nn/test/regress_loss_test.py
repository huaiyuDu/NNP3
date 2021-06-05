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
data_train = pd.read_csv('./../data/classification/data.three_gauss.train.100.csv')
data_test = pd.read_csv('./../data/classification/data.three_gauss.test.100.csv')
(x_train, y_train, x_test, y_test) = pre.prepare_data_train_test(data_train, data_test,one_hot_encoding=True)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# hyper parameters
max_loops = 3000
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
layers.append(Layer(2, None))
layers.append(Layer(4, ReluFunction()))
layers.append(Layer(3, SoftMaxFunction()))

figure, axis = plt.subplots(3, num_per_case)
figure_bounary, axis_bounary = plt.subplots(3, num_per_case)
for i in range(0,num_per_case):

    mpl = MultilayerPerceptron(layers,CrossEntropy(), weight_scale=0.5)
    (cost, train_precision, test_precision)=et.exec_multipl_cls(mpl,x_train,y_train,x_test, y_test,max_loops,batch_size, alpha)
    mpl_costs.append(cost)
    mpl_precisions.append(("layer1-" + str(i), train_precision,test_precision))
    plot.plot_cost(cost,axis, 0,i  ,"layer1-" + str(i))
    mpls.append(mpl)
    # if i == 0:
    #     plot.plot_multipl_decision_region(x_train[:,0:100],y_train[:,0:100],3,mpl)
    plot.plot_multipl_decision_region_axis(x_train[:,0:100],y_train[:,0:100],3,mpl,axis_bounary, 0,i,"layer1-"+ str(i))

# sigmoid
layers = []
layers.append(Layer(2, None))
layers.append(Layer(4, ReluFunction()))
layers.append(Layer(4, ReluFunction()))
layers.append(Layer(3, SoftMaxFunction()))
for i in range(0,num_per_case):

    mpl = MultilayerPerceptron(layers,CrossEntropy(),weight_scale=0.1)
    (cost, train_precision, test_precision)=et.exec_multipl_cls(mpl,x_train,y_train,x_test, y_test,max_loops,batch_size, alpha)
    mpl_costs.append(cost)
    mpl_precisions.append(("layer2-" + str(i), train_precision,test_precision))
    plot.plot_cost(cost,axis, 1,i,"layer2-" + str(i))
    mpls.append(mpl)
    # if i == 0:
    #     plot.plot_multipl_decision_region(x_train[:,0:100],y_train[:,0:100],3,mpl)
    plot.plot_multipl_decision_region_axis(x_train[:,0:100],y_train[:,0:100],3,mpl,axis_bounary, 1,i,"layer2-"+ str(i))
#
# # tanh
layers = []
layers.append(Layer(2, None))
layers.append(Layer(4, ReluFunction()))
layers.append(Layer(6, ReluFunction()))
layers.append(Layer(4, ReluFunction()))
layers.append(Layer(3, SoftMaxFunction()))
for i in range(0,num_per_case):

    mpl = MultilayerPerceptron(layers,CrossEntropy(),weight_scale=0.3)
    (cost, train_precision, test_precision)=et.exec_multipl_cls(mpl,x_train,y_train,x_test, y_test,max_loops,batch_size, alpha)
    mpl_costs.append(cost)
    mpl_precisions.append(("layer3-" + str(i), train_precision,test_precision))
    plot.plot_cost(cost,axis, 2,i,"layer3-" + str(i))
    # if i == 0:
    #     plot.plot_multipl_decision_region(x_train[:,0:100],y_train[:,0:100],3,mpl)
    plot.plot_multipl_decision_region_axis(x_train[:,0:100],y_train[:,0:100],3,mpl,axis_bounary, 2,i,"layer3-"+ str(i))
    #print(cost)
    mpls.append(mpl)


plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.show()
for i in range (0,3*num_per_case,num_per_case):
    plot.plot_multipl_decision_region(x_train[:,0:100],y_train[:,0:100],3,mpls[i])
print(mpl_precisions)