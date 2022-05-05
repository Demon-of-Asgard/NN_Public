import inspect
import numpy as np
import matplotlib.pyplot as plt
from terminal_colors import term_colors as tcl


import model
import data_handler as dhand


def image_plot(image, text):
    plt.imshow(image)
    plt.title(text)
    plt.show()


def plot_cost(cost):
    plt.plot(cost['iteration'], cost['cost'])
    plt.xlabel("Iteration")
    plt.ylabel("cost")
    plt.show()


def main():

    norm = 1.0/255.0

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = dhand.load_data()
    X_train, Y_train = norm*dhand.cast_xdata(X_train_orig), dhand.cast_ydata(Y_train_orig)

    assert X_train.shape[1] == Y_train.shape[1], f"{tcl.FAIL}Dimension missmatch. {X_train.shape[1]}!={Y_train.shape[1]} module: {inspect.stack()[0][3]}, file: {__name__}, called from: {inspect.stack()[1][3]}.{tcl.ENDC}"
    print(f"Training model ... ")
    cost, params = model.model(X_train, Y_train)
    plot_cost(cost)

    w = params["w"]
    b = params["b"]

    while True:
        id = int(input(f"Enter an image id [0-{Y_test_orig.size-1}/{-1} to exit]: "))
        if id == -1:
            break
        else:
            prediction = ""
            X_test = norm*dhand.cast_xdata(X_test_orig[id]).reshape(-1, 1)
            prediction = model.predict(X_test, w, b)
            image_plot(image=X_test_orig[id], text=prediction)


if __name__ == "__main__":
    main()