import sys
import numpy as np

def sigmoid(arr:np.ndarray)->np.ndarray:
    return 1.0/(1+np.exp(-arr))



