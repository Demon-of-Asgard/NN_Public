
import sys
import numpy as np
from sigmoid import sigmoid
from load_data import load_data
from terminal_colors import term_colors as tcl


if __name__ == "__main__":
    result = sigmoid(np.array([1,2,0])).reshape(3, 1)
    expected_result = np.array([
        [0.73105858],
        [0.88079708],
        [0.5      ],
        ])
    assert np.allclose(result, expected_result), f"{tcl.FAIL} expect {expected_result}. You got {result}{tcl.ENDC}"
    print(f"{tcl.OKGREEN}Sigmoid test passed {tcl.ENDC}")
    
    train_x, train_y, test_x, test_y, classes = load_data()

    assert train_x.shape[0] == train_y.shape[0], f"{tcl.FAIL}#  Ttain data size does not match with # train y data size."

    print(f"Train_x: {train_x.shape} \nTest_x: {test_x.shape}")