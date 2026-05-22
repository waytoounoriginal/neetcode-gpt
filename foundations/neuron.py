import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def relu(x):
            return max(0, x)

        activator = sigmoid if activation == "sigmoid" else relu

        y_hat = float(activator(np.dot(w, x) + b))
        return round(y_hat, 5)
