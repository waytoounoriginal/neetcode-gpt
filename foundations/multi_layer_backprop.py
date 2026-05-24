import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        # Convert inputs to arrays so matrix ops (and W2.T) work
        x      = np.asarray(x,      dtype=float)   # (input,)
        W1     = np.asarray(W1,     dtype=float)   # (hidden, input)
        b1     = np.asarray(b1,     dtype=float)   # (hidden,)
        W2     = np.asarray(W2,     dtype=float)   # (output, hidden)
        b2     = np.asarray(b2,     dtype=float)   # (output,)
        y_true = np.asarray(y_true, dtype=float)   # (output,)

        # n = number of output dimensions (single training example)
        n = len(y_true)

        # ---- Forward pass ----
        z1 = np.matmul(W1, x) + b1   # (hidden, input) @ (input,) + (hidden,) -> (hidden,)
        a1 = np.maximum(z1, 0)       # ReLU activation                        -> (hidden,)

        y_hat = np.matmul(W2, a1) + b2  # (output, hidden) @ (hidden,) + (output,) -> (output,)

        # Mean squared error over the output dimensions
        loss = np.mean((y_hat - y_true) ** 2)

        # ---- Backward pass ----
        # dL/dy_hat for loss = mean((y_hat - y_true)^2); the 2/n comes from d/dx of mean of squares
        dl_dy_hat = 2 / n * (y_hat - y_true)   # (output,)

        # Layer 2 gradients: y_hat = W2 @ a1 + b2
        dl_dW2 = np.outer(dl_dy_hat, a1)       # outer(dL/dy_hat, a1) -> (output, hidden)
        dl_db2 = dl_dy_hat                     # bias gradient is just dL/dy_hat -> (output,)

        # Backprop into hidden layer, then through the ReLU
        # dL/dz1 = (W2^T @ dL/dy_hat) elementwise-* ReLU'(z1), where ReLU'(z1) = (z1 > 0)
        dl_dz1 = np.matmul(W2.T, dl_dy_hat) * (z1 > 0)   # (hidden, output) @ (output,) -> (hidden,)

        # Layer 1 gradients: z1 = W1 @ x + b1
        dl_dW1 = np.outer(dl_dz1, x)
        # Ensure we don't have negative zeros in the output due to rounding floating point zeros
        dl_dW1 = np.where(dl_dW1 == 0, 0, dl_dW1)
        dl_db1 = np.where(dl_dz1 == 0, 0, dl_dz1)
        dl_dW2 = np.where(dl_dW2 == 0, 0, dl_dW2)
        dl_db2 = np.where(dl_db2 == 0, 0, dl_db2)

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dl_dW1, 4).tolist(),
            'db1': np.round(dl_db1, 4).tolist(),
            'dW2': np.round(dl_dW2, 4).tolist(),
            'db2': np.round(dl_db2, 4).tolist(),
        }