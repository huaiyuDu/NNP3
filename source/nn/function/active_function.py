from abc import ABCMeta, abstractmethod
import numpy as np

big_number = 10000

class ActiveFunction:
    __metaclass__ = ABCMeta

    @abstractmethod
    def g(self, z): raise NotImplementedError

    @abstractmethod
    def derivative_g(self, a): raise NotImplementedError

    @abstractmethod
    def derivative_g_quick(self, a, y): raise NotImplementedError


class SigmoidFunction(ActiveFunction):
    # def g(self, z):
    #     e_z = np.exp(-z)
    #     e_z[e_z == np.inf] = big_number
    #     #return 1 / (1 + np.exp(-z))
    #     return 1 / (1 + e_z)
    def g(self, z):
        #e_z = np.exp(-z)
        #e_z[e_z == np.inf] = big_number
        return 1 / (1 + np.exp(-z))
    def derivative_g(self, a):
        sigm_a = self.g(a)
        return sigm_a * (1 - sigm_a)

    def derivative_g_quick(self, a, y): raise NotImplementedError

class TanhFunction(ActiveFunction):
    def g(self, z):
        #print('z')
        #print(z)
        e_a_z = np.exp(z)
        e_a_z[e_a_z == np.inf] = big_number
        e_m_z = np.exp(-z)
        e_m_z[e_m_z == np.inf] = big_number
        return (e_a_z -e_m_z) / (e_a_z +e_m_z)

    def derivative_g(self, a):
        #print('a')
        #print(a)
        return 1 - np.power(self.g(a), 2)

    def derivative_g_quick(self, a, y): raise NotImplementedError


class ReluFunction(ActiveFunction):
    def g(self, z):
        return np.maximum(z, 0)

    def derivative_g(self, a):
        return np.where(a <= 0, 0, 1)

    def derivative_g_quick(self, a, y): raise NotImplementedError

class LeakyReluFunction(ActiveFunction):
    def g(self, z):
        return  np.where(z > 0, z, z * 0.01)

    def derivative_g(self, a):
        return np.where(a <= 0, 0.01, 1)

    def derivative_g_quick(self, a, y): raise NotImplementedError


class SoftMaxFunction(ActiveFunction):
    def g(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    def derivative_g(self, a):
        raise NotImplementedError

    def derivative_g_quick(self, a, y):
        return a - y

class LinnerFunction(ActiveFunction):
    def g(self, z):
        return z

    def derivative_g(self, a):
        return np.ones(a.shape)

    def derivative_g_quick(self, a, y):
        raise NotImplementedError

# test
# a= np.array([[-1 , 3 , 5]])
# print(a)
# f = ReluFunction()
# print(f.g(a))
# print(f.derivative_g(a))
import numpy as np

def softmax_grad(s):
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x.
    # s.shape = (1, n)
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])

    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(s)

    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else:
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m

s = np.array([0.3, 0.7])
sd = softmax_grad(s)
print(sd)