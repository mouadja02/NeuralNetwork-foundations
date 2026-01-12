"""
Train a neural network to learn XOR function.

XOR is not linearly separable, so we need:
- At least one hidden layer
- Non-linear activation function

Network architecture: 2 -> 4 -> 1
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.matrix import Matrix
from core.layer import DenseLayer
from core.activations import sigmoid, sigmoid_derivative
from core.loss import mean_squared_error, mse_derivative
from utils.progress import ProgressBar


# XOR dataset
X_train = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y_train = [
    [0],
    [1],
    [1],
    [0]
]


def train_xor():
    """Train a 2-layer network to learn XOR."""

    print("\n" + "="*60)
    print(" 🧠 TRAINING XOR NETWORK")
    print("="*60 + "\n")

    # Create network: 2 -> 4 -> 1
    layer1 = DenseLayer(2, 4)
    layer2 = DenseLayer(4, 1)

    # Hyperparameters
    epochs = 10000
    learning_rate = 0.5

    # Training loop
    pbar = ProgressBar(epochs, desc="Training")

    for epoch in range(epochs):
        total_loss = 0

        # Train on each sample
        for i in range(len(X_train)):
            X = Matrix([X_train[i]])
            y_true = Matrix([y_train[i]])

            # Forward pass
            # 1. h1 = layer1.forward(X)
            h1 = layer1.forward(X)
            a1 = sigmoid(h1)
            h2 = layer2.forward(a1)
            y_pred = sigmoid(h2)
            
            # Compute loss
            loss = mean_squared_error(y_true, y_pred)
            total_loss += loss

            # Backward pass
            # Chain rule: dL/dh2 = dL/dy_pred * dy_pred/dh2

            # 1. Compute gradient of loss with respect to predictions
            dL_dy_pred = mse_derivative(y_true, y_pred)

            # 2. Gradient through sigmoid activation (y_pred = sigmoid(h2))
            # sigmoid_derivative expects the OUTPUT of sigmoid (y_pred)
            dy_pred_dh2 = sigmoid_derivative(y_pred)

            # Element-wise multiply to get dL/dh2
            dL_dh2 = Matrix([[dL_dy_pred.data[i][j] * dy_pred_dh2.data[i][j]
                             for j in range(dL_dy_pred.cols)]
                            for i in range(dL_dy_pred.rows)])

            # 3. Backprop through layer2
            dL_da1 = layer2.backward(dL_dh2, learning_rate)

            # 4. Gradient through sigmoid activation (a1 = sigmoid(h1))
            da1_dh1 = sigmoid_derivative(a1)

            # Element-wise multiply to get dL/dh1
            dL_dh1 = Matrix([[dL_da1.data[i][j] * da1_dh1.data[i][j]
                             for j in range(dL_da1.cols)]
                            for i in range(dL_da1.rows)])

            # 5. Backprop through layer1
            layer1.backward(dL_dh1, learning_rate)

        # Update progress bar every 100 epochs
        if epoch % 100 == 0:
            avg_loss = total_loss / len(X_train)
            pbar.set_description(f"Training | Loss: {avg_loss:.6f}")

        pbar.update(1)

    pbar.close()

    # Test the network
    print("\n" + "="*60)
    print(" 🧪 TESTING XOR NETWORK")
    print("="*60 + "\n")

    print("Input  | Predicted | True | Correct?")
    print("-" * 40)

    correct = 0
    for i in range(len(X_train)):
        X = Matrix([X_train[i]])

        # Forward pass
        h1 = layer1.forward(X)
        a1 = sigmoid(h1)
        h2 = layer2.forward(a1)
        y_pred = sigmoid(h2)

        predicted_value = y_pred.data[0][0]
        true_value = y_train[i][0]

        # Round to 0 or 1
        predicted_class = 1 if predicted_value > 0.5 else 0

        is_correct = (predicted_class == true_value)
        if is_correct:
            correct += 1

        print(f"{X_train[i]} | {predicted_value:.4f}    | {true_value}    | {'✓' if is_correct else '✗'}")

    accuracy = correct / len(X_train) * 100
    print("-" * 40)
    print(f"Accuracy: {accuracy:.1f}%")

    if accuracy >= 95:
        print("\n🎉 SUCCESS! Your network learned XOR!")
    else:
        print("\n❌ Network didn't converge. Try adjusting hyperparameters.")

    print("="*60 + "\n")


if __name__ == "__main__":
    train_xor()