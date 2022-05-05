import inspect
import numpy as np
from sigmoid import sigmoid
from terminal_colors import term_colors as tcl

def propagate(X, Y, w, b):
    assert Y.shape[0] == 1, f"{tcl.FAIL}Y is not a column vector here. Y has shape. {Y.shape}. Track here :: module: <{inspect.stack()[0][3]}>, file: <{__name__}>, called from: <{inspect.stack()[1][3]}>.{tcl.ENDC}"
    assert X.shape[0] == w.shape[0], f"{tcl.FAIL}Dimension mismatch. Size of axis-0 of X and w are not same. Track here :: module: <{inspect.stack()[0][3]}>, file: <{__name__}>, called from: <{inspect.stack()[1][3]}>.{tcl.ENDC}."
    assert X.shape[1] == Y.shape[1], f"{tcl.FAIL}Dimension mismatch. Size of axis-1 of X and Y are not same. Track here :: module: <{inspect.stack()[0][3]}>, file: <{__name__}>, called from: <{inspect.stack()[1][3]}>.{tcl.ENDC}"

    m = Y.shape[1]

    z = w.T@X + b # (n_feature, m_sample).T * (n_feature, 1)  --> (m_sample, 1) + b
    prediction = sigmoid(z)

    cost = -(1.0/m) * ( Y@np.log(prediction).T + (1 - Y)@np.log(1 - prediction).T)

    dw = (1.0/m) * X@(prediction - Y).T
    db = (1.0/m) * (prediction - Y).sum()

    grad = {"dw" : dw, "db" : db}

    return cost, grad


def optimize(X, Y, w, b, learning_rate=0.01, niter = 10000):

    cost = {'iteration' : [], 'cost' : []}
    
    for i in range(niter):
        cost_i, grad = propagate(X, Y, w, b)
        dw = grad["dw"]
        db = grad["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db
        

        if i%100 == 0:
            cost['iteration'].append(i)
            cost['cost'].append(cost_i[0][0])
            print(f"Cost after {i} iterations: {cost_i}")


    params = {"w" : w, "b" : b}

    return cost, params

