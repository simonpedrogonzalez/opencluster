import numpy as np

def subset(data: np.ndarray, limits: list):
    for i in range(len(limits)):
        data = data[(data[:,i] > limits[i][0]) & (data[:,i] < limits[i][1])]
    return data