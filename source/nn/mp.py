import numpy as np
from nn.function.active_function import SoftMaxFunction

class MultilayerPerceptron:

    def __init__(self, X, Y, layers, loss_function, task_type='classification' ):

        self.X = X
        self.Y = Y
        self.layers = layers
        self.task_type = task_type
        self.loss_function = loss_function
        np.random.seed(2)
        (weights, biases) = self.init_weight_bias(layers)
        self.weights = weights
        self.biases = biases
        self.delta_weight_his = []
        self.delta_b_his = []


    def init_weight_bias(self, layers):
        num_layers = len(layers)
        weights = {}
        biases = {}
        for layer_index in range(1, num_layers ):
            col_num = layers[layer_index -1].neural_number
            row_num = layers[layer_index].neural_number
            weights[layer_index] = np.random.rand(row_num, col_num)*0.5
            biases[layer_index] = np.random.rand(row_num, 1)*0.5
        return weights, biases

    def train(self , max_times, batch_size, alpha , adma = 0):
        #np.array_split(self.X, )
        num_test_case = self.X.shape[1]
        loop_times = 0
        costs = []
        for i in range (max_times):
            for start in range(0, num_test_case, batch_size):
                x_train = self.X[:, start:start+batch_size]
                y_train = self.Y[:, start:start+batch_size]
                cost = self.forward_backward_train(x_train, y_train, self.layers, self.weights, self.biases, alpha , adma)
                costs.append(cost)
                if loop_times%100 == 0:
                    print("loop "+ str(loop_times))
                #print(LA.norm(self.weights))
                #print(self.biases)
                # start forward and backward
                loop_times += 1
                start = start + batch_size

        return costs

    def test(self , test_x):
        #np.array_split(self.X, )
        (predict_y, z_catch,a_catch)= self.forward_propogation(test_x, self.layers, self.weights, self.biases, False)
        return predict_y

    def forward_backward_train(self, train_x, train_y, layers, weights, biases, alpha, adma = 0):
        (predict_y, z_catch,a_catch )= self.forward_propogation(train_x, layers, weights, biases, True)
        cost = self.back_propogation(train_x, train_y, layers, weights, biases, z_catch,a_catch, alpha,adma)
        return cost;

    def forward_propogation(self, train_x, layers, weights, biases, catch_result = False):

        # A_0 = x
        # Z_{i+1} = W*A_i +B
        # A_{i+1} = g(Z_{i+1}), g(x) is active function
        z_catch = {}
        a_catch = {}
        num_layers = len(layers)

        a_in = train_x
        a_catch[0] = a_in
        for current_layer in range(1, num_layers):
            #print("current_layer"+str(current_layer))
            #print(weights[current_layer].shape)
            z_out = np.dot(weights[current_layer],a_in) + biases[current_layer]
            a_out = layers[current_layer].active_function.g(z_out)
            if catch_result:
                z_catch[current_layer] = z_out
                a_catch[current_layer] = a_out
            a_in = a_out
        predict_y = a_in
        return predict_y, z_catch, a_catch

    def back_propogation(self,  train_x, train_y,  layers, weights, biases, z_catch,a_catch, alpha , adam = 0):
        num_layers = len(layers)
        num_examples = train_x.shape[1]
        d_a = self.loss_function.derivative_loss(train_y, a_catch[num_layers - 1])
        cost = 1/ num_examples* np.sum(self.loss_function.loss(train_y, a_catch[num_layers - 1]))
        weight_his_record = {}
        b_his_record = {}
        hasLast = False
        if len(self.delta_weight_his)> 0:
            hasLast = True
            last_weight_his_rocord = self.delta_weight_his[len(self.delta_weight_his)-1]
            last_b_his_rocord = self.delta_b_his[len(self.delta_b_his)-1]

        for layer_index in range(num_layers -1 , 0 , -1):
            z = z_catch[layer_index]
            if isinstance(layers[layer_index].active_function,SoftMaxFunction):
                d_z = layers[layer_index].active_function.derivative_g_quick(a_catch[num_layers - 1],train_y)
            else:
                d_z = d_a * layers[layer_index].active_function.derivative_g(z)
            d_w = 1/num_examples * np.dot(d_z, a_catch[layer_index-1].T)
            d_b = 1/num_examples * np.sum(d_z, axis=1 , keepdims=True)
            # update weight
            if hasLast and adam > 0:
                d_w = d_w + adam*last_weight_his_rocord[layer_index]
                d_b = d_b + adam*last_b_his_rocord[layer_index]
            weight_his_record[layer_index] = d_w
            b_his_record[layer_index] = d_b
            weights[layer_index] = weights[layer_index] - alpha * d_w
            biases[layer_index] = biases[layer_index] - alpha * d_b
            if layer_index > 1:
                d_a =  np.dot(weights[layer_index].T,d_z)
        #print("w_norm = "+str(LA.norm(d_w))+ ", b_norm = " +str(LA.norm(d_b)))
        self.delta_weight_his.append(weight_his_record)
        self.delta_b_his.append(b_his_record)
        return cost

class Layer:
    def __init__(self, neural_number, active_function):
        self.neural_number = neural_number
        self.active_function = active_function
