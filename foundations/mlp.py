import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        curr_answer = x # (in,)
        for i in range(len(biases) - 1):
            curr_answer = np.matmul(curr_answer, weights[i]) + biases[i]
            curr_answer = np.maximum(curr_answer, 0) # relu activation

        y_hat = curr_answer @ weights[-1] + biases[-1]
        return np.round(y_hat, 5)
