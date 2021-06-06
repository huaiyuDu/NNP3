from abc import ABCMeta, abstractmethod
import numpy as np

tolerance = 10**-15
class ILossFunction:


    __metaclass__ = ABCMeta

    @abstractmethod
    def loss(self, y, a): raise NotImplementedError

    @abstractmethod
    def derivative_loss(self,y, a): raise NotImplementedError


class BinaryCrossEntropy(ILossFunction):
    def loss(self, y, a):
        n = y.shape[0]
        return - 1/n * np.sum(y * np.log(a) + (1-y) * np.log(1-a), axis=0, keepdims= True)

    def derivative_loss(self, y, a):
        a[np.isclose(a, 0, atol=tolerance)] = tolerance
        a[np.isclose(a, 1, atol=tolerance)] = 1-tolerance
        return np.divide(-y, a) + np.divide((1 - y), (1-a))


class CrossEntropy(ILossFunction):
    def loss(self, y, a):
        n = y.shape[0]
        return -np.sum(y * np.log(a), axis=0, keepdims= True)

    def derivative_loss(self, y, a):
        a[np.isclose(a, 0, atol=tolerance)] = tolerance
        a[np.isclose(a, 1, atol=tolerance)] = 1-tolerance
        return np.divide(-y, a)


class MeanSquaredError(ILossFunction):
    def loss(self, y, a):
        return np.sum(np.power(y-a,2), axis=0, keepdims= True)/y.shape[0]

    def derivative_loss(self, y, a):

        return 2*(a - y)/ y.shape[0]

class MeanAbsoluteError(ILossFunction):
    def loss(self, y, a):
        return np.sum(abs(y-a), axis=0, keepdims= True)/y.shape[0]

    def derivative_loss(self, y, a):

        return np.where(y > a, -1, +1)/ y.shape[0]

#
#lossF = BinaryCrossEntropy()
# y = np.array([[1 , 1],
#               [0, 1]])
# a = np.array([[0.5, 0.1],
#               [0.1 , 0.9]])
y = np.array([[1 , 1]])
a = np.array([[0.1 , 0.9]])


print(np.where(y >= a, 1, -1))
