import sys
import sigmoid
import inspect
import numpy as np
from optimize import optimize 
from terminal_colors import term_colors as tcl



def predict(X, w, b):
    prediction = -1.0
    prediction = sigmoid.sigmoid(w.T@X + b)
    if prediction >= 0.5 and prediction <= 1.0:
        return "Its a cat"
    elif prediction >= 0 and prediction < 0.5:
        return "it's not a cat"
    else:
        return "I am confused"



def model(X_train, Y_train):
    n_features = X_train.shape[0]
    m = X_train.shape[1]
    assert m == Y_train.shape[1], f"{tcl.FAIL}shape missmatch. Y_train.shape[0] != X_train.shape[0] :: module: <{inspect.stack()[0][3]}>, file: <{__name__}>, called from: <{inspect.stack()[1][3]}>.{tcl.ENDC}"
    w = np.zeros((n_features, 1))
    b = np.zeros((1,1))
    print(f"Optimizing ... ")
    cost, grad = optimize(X_train, Y_train, w, b)
    return cost, grad

   