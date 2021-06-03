import numpy as np
def expend_to_polynomial(x,degree):
    result = np.copy(x).astype(float)
    for i in range(2, degree+1):
        px = np.power(x, i)
        result = np.concatenate((result,px), axis=0)
    return result

# test
# x = np.array([[1,2 ,3]])
# #x = x.reshape((1,x.shape[0])).T
# x = x.T
# print(x)
# print(expend_to_polynomial(x,3))
