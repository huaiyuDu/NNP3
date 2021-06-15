import numpy as np
import pandas as pd
def to_one_hot(y_labels):
    df = pd.DataFrame(y_labels)
    df_transformed = pd.get_dummies(df[0])
    return df_transformed.values


def one_hot_encoding(y, n):
    count = y.shape[0]
    return np.eye(n)[y].reshape(count,n)
y = np.array([[0, 3 , 2 , 1 ,1 ,2 ,3]])

# y_encoding = one_hot_encoding(y,4)
# print(y_encoding)
# print(np.argmax(y_encoding.T,axis=0))
