import math
from core.matrix import Matrix, zeros, ones, identity
import random


class DenseLayer:
    """
    Fully connected (dense) neural network layer.

    Architecture:
    Input: (batch_size, input_size)
    Output: (batch_size, output_size)

    Parameters:
    - W: (input_size, output_size) weight matrix
    - b: (output_size,) bias vector

    Example:
        layer = DenseLayer(784, 128)  # MNIST input -> hidden layer
        output = layer.forward(X)      # X shape: (32, 784)
                                       # output shape: (32, 128)
    """

    def __init__(self, input_size, output_size):
        """
        Initialize layer with Xavier/Glorot initialization.

        Xavier initialization: weights ~ N(0, sqrt(2/(input + output)))
        This helps prevent vanishing/exploding gradients.

        Args:
            input_size: Number of input features
            output_size: Number of output features (neurons)
        """
        # Xavier initialization
        scale = math.sqrt(2.0/ (input_size + output_size))
        self.W = Matrix([[scale * random.random() * 2 - 1 for _ in range(output_size)] for _ in range(input_size)])
        self.b = zeros(1, output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.X = None 

    def forward(self, X):
        """
        Forward pass: compute Y = X @ W + b

        Args:
            X: Input matrix (batch_size, input_size)
               Can be Matrix object or list of lists

        Returns:
            Y: Output matrix (batch_size, output_size)

        Example:
            X = Matrix([[1, 2, 3],
                        [4, 5, 6]])  # 2 samples, 3 features
            layer = DenseLayer(3, 2)
            Y = layer.forward(X)     # Shape: (2, 2)
        """
        # 1. Convert X to Matrix if it's a list
        if isinstance(X, list):
            X = Matrix(X)
        self.X = X
        dot_result = X.dot(self.W)
        if dot_result.shape() == (X.rows, self.output_size) and self.b.shape() == (1, self.output_size):
            return dot_result + ones(X.rows, 1).dot(self.b)
        else:
            return dot_result + self.b

    def backward(self, dL_dY, learning_rate):
        """
        Backward pass: compute gradients and update weights.

        Given the gradient of loss with respect to output (dL/dY),
        compute:
        1. Gradient with respect to weights: dL/dW = X^T @ dL/dY
        2. Gradient with respect to biases: dL/db = sum(dL/dY, axis=0)
        3. Gradient with respect to input: dL/dX = dL/dY @ W^T

        Then update weights:
        W = W - learning_rate * dL/dW
        b = b - learning_rate * dL/db

        Args:
            dL_dY: Gradient from next layer (batch_size, output_size)
            learning_rate: Step size for gradient descent

        Returns:
            dL_dX: Gradient to pass to previous layer (batch_size, input_size)

        Example:
            dL_dY = Matrix([[0.1, 0.2],
                            [0.3, 0.4]])  # Gradient from next layer
            dL_dX = layer.backward(dL_dY, learning_rate=0.01)
        """
        # 1. Convert dL_dY to Matrix if needed
        if not isinstance(dL_dY, Matrix):
            dL_dY = Matrix(dL_dY)
        # 2. Compute dL/dW = X^T @ dL/dY
        dL_dW = self.X.T().dot(dL_dY)
        # 3. Compute dL/db = sum of dL/dY along axis=0
        dL_db = []
        for i in range(dL_dY.cols):
            sum = 0
            for j in range(dL_dY.rows):
                sum += dL_dY.data[j][i]
            dL_db.append(sum)
        dL_db = Matrix([dL_db])
        # 4. Compute dL/dX = dL/dY @ W^T
        dL_dX = dL_dY.dot(self.W.T())
        # 5. Update weights:
        def update_weights(x):
            return -learning_rate * x
        self.W = self.W + dL_dW.apply(update_weights)
        self.b = self.b + dL_db.apply(update_weights)
        # 6. Return dL/dX
        return dL_dX

    def get_params(self):
        """
        Get current parameters (for inspection/debugging).

        Returns:
            Dictionary with 'W' and 'b' keys
        """
        return {'W': self.W, 'b': self.b}

    def __repr__(self):
        return f"DenseLayer({self.input_size} -> {self.output_size})"