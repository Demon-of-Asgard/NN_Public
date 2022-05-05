import h5py
import numpy as np


def load_data():
    train_data = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_x = np.array(train_data["train_set_x"][:])
    train_y = np.array(train_data["train_set_y"][:])

    test_data = test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_x = np.array(test_data["test_set_x"][:])
    test_y =  np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    return train_x, train_y, test_x, test_y, classes


def cast_xdata(data):
    return data.reshape(data.shape[0], -1).T



def cast_ydata(data):
    return data.reshape(1, data.shape[0])