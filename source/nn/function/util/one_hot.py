import numpy as np
import pandas as pd
def to_one_hot(y_labels):
    df = pd.DataFrame(y_labels.T)
    df_transformed = pd.get_dummies(df[0])
    return df_transformed.values.T

# y = np.array([[ 3 , 2 , 1 ,1 ,2 ,3]])
# print(to_one_hot(y))