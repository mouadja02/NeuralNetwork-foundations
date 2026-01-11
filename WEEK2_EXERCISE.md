# Week 2 - Exercises 1.2 & 1.3: Activation Functions & Loss Functions

Congratulations on completing Week 1! Now we'll implement the mathematical functions that make neural networks work.

---

## 🎯 Overview

This week you'll build:
1. **Activation functions** - The non-linear transformations that give neural networks their power
2. **Loss functions** - How we measure how wrong our predictions are

These are the mathematical heart of neural networks!

---

## 📝 Exercise 1.2: Activation Functions

### Why Activation Functions?

Without activation functions, a neural network is just a linear transformation (matrix multiply). No matter how many layers you stack, it's equivalent to a single matrix multiply!

**Key insight**: Activation functions add **non-linearity**, allowing networks to learn complex patterns.

### Your Task

Create `python/core/activations.py` and implement these functions:

```python
"""
Activation functions and their derivatives.

Each function works on:
- Scalars (single numbers)
- Lists (element-wise)
- Matrix objects (element-wise)
"""

import math


def sigmoid(x):
    """
    Sigmoid activation: maps input to (0, 1)

    Formula: σ(x) = 1 / (1 + e^(-x))

    Properties:
    - Output range: (0, 1)
    - Smooth, differentiable
    - Problem: vanishing gradients for |x| > 5

    Use: Binary classification, gates in LSTMs

    Args:
        x: number, list, or Matrix

    Returns:
        Same type as input, with sigmoid applied element-wise

    Examples:
        sigmoid(0) = 0.5
        sigmoid(large positive) ≈ 1
        sigmoid(large negative) ≈ 0
    """
    # TODO: Implement
    # Hints:
    # 1. Check if x is a number, list, or Matrix
    # 2. For numbers: return 1 / (1 + math.exp(-x))
    # 3. For lists: apply recursively
    # 4. For Matrix: use the apply() method you wrote!
    pass


def sigmoid_derivative(x):
    """
    Derivative of sigmoid.

    Formula: σ'(x) = σ(x) * (1 - σ(x))

    This is why sigmoid is convenient - its derivative is simple!

    Args:
        x: number, list, or Matrix (can be pre-computed sigmoid values)

    Returns:
        Derivative at each point

    Note: This function expects the OUTPUT of sigmoid (not the input).
    So you'd call it like: sigmoid_derivative(sigmoid(x))
    """
    # TODO: Implement
    # If x is already the sigmoid output: return x * (1 - x)
    pass


def relu(x):
    """
    ReLU (Rectified Linear Unit): most popular activation

    Formula: f(x) = max(0, x)

    Properties:
    - Output range: [0, ∞)
    - Dead simple, very fast
    - No vanishing gradient for x > 0
    - Problem: "dead neurons" (always output 0)

    Use: Hidden layers in most modern networks

    Examples:
        relu(-5) = 0
        relu(0) = 0
        relu(5) = 5
    """
    # TODO: Implement
    # Hint: Use max(0, x) for scalars
    pass


def relu_derivative(x):
    """
    Derivative of ReLU.

    Formula: f'(x) = 1 if x > 0 else 0

    Technically undefined at x=0, but we use 0 by convention.

    Note: This expects the INPUT to ReLU (not the output).
    """
    # TODO: Implement
    pass


def tanh(x):
    """
    Hyperbolic tangent: zero-centered sigmoid

    Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

    Properties:
    - Output range: (-1, 1)
    - Zero-centered (better than sigmoid)
    - Still suffers from vanishing gradients

    Use: RNNs, when you want zero-centered output

    Python has math.tanh() but implement it yourself!
    """
    # TODO: Implement using the formula
    pass


def tanh_derivative(x):
    """
    Derivative of tanh.

    Formula: tanh'(x) = 1 - tanh²(x)

    Note: Expects the OUTPUT of tanh.
    """
    # TODO: Implement
    pass


def softmax(x):
    """
    Softmax: converts logits to probability distribution

    Formula: softmax(x_i) = e^(x_i) / Σ(e^(x_j))

    Properties:
    - Outputs sum to 1
    - Each output in (0, 1)
    - Differentiable

    Use: Multi-class classification (output layer)

    IMPORTANT: Numerical stability!
    e^x explodes for large x. Use this trick:
    softmax(x) = softmax(x - max(x))

    This doesn't change the result but prevents overflow.

    Args:
        x: list or Matrix (treats as vector or row vectors)

    Returns:
        Same shape, normalized to probability distribution

    Examples:
        softmax([1, 2, 3]) ≈ [0.09, 0.24, 0.67]
        Notice: sum = 1.0
    """
    # TODO: Implement with numerical stability
    # Steps:
    # 1. Subtract max(x) from all elements
    # 2. Compute e^x for each element
    # 3. Divide each by the sum
    pass


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU: fixes "dead neurons" problem

    Formula: f(x) = x if x > 0 else alpha * x

    Properties:
    - Allows small gradient when x < 0
    - No dead neurons
    - alpha typically 0.01

    Use: Alternative to ReLU in hidden layers
    """
    # TODO: Implement (bonus!)
    pass


def leaky_relu_derivative(x, alpha=0.01):
    """
    Derivative of Leaky ReLU.

    Formula: f'(x) = 1 if x > 0 else alpha
    """
    # TODO: Implement (bonus!)
    pass


# Helper function for type checking
def _handle_types(func, x):
    """
    Helper to apply activation function to different types.

    You can use this in your implementations!
    """
    from core.matrix import Matrix

    if isinstance(x, Matrix):
        # Use the apply() method from your Matrix class!
        return x.apply(func)
    elif isinstance(x, list):
        # Recursively handle nested lists
        return [_handle_types(func, xi) for xi in x]
    else:
        # Assume it's a number
        return func(x)
```

---

## 📝 Exercise 1.3: Loss Functions

### Why Loss Functions?

The loss function tells us "how wrong" our predictions are. During training, we try to minimize this loss.

Different tasks need different loss functions:
- **Regression**: MSE (Mean Squared Error)
- **Binary classification**: Binary Cross-Entropy
- **Multi-class classification**: Categorical Cross-Entropy

### Your Task

Create `python/core/loss.py`:

```python
"""
Loss functions and their derivatives.

These measure how wrong our predictions are.
"""

import math


def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error: for regression tasks

    Formula: MSE = (1/n) * Σ(y_true - y_pred)²

    Properties:
    - Always non-negative
    - Penalizes large errors heavily (squared)
    - Differentiable everywhere

    Use: Regression (predicting continuous values)

    Args:
        y_true: list or Matrix of true values
        y_pred: list or Matrix of predicted values

    Returns:
        Single number (the loss)

    Examples:
        mse([1,2,3], [1,2,3]) = 0.0  (perfect)
        mse([0,0], [1,1]) = 1.0
    """
    # TODO: Implement
    # Steps:
    # 1. Handle Matrix/list inputs
    # 2. Compute differences: (y_true - y_pred)
    # 3. Square them
    # 4. Take mean
    pass


def mse_derivative(y_true, y_pred):
    """
    Derivative of MSE with respect to predictions.

    Formula: dMSE/dy_pred = (2/n) * (y_pred - y_true)

    This tells us which direction to adjust predictions.

    Returns:
        Same shape as inputs
    """
    # TODO: Implement
    pass


def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Binary Cross-Entropy: for binary classification

    Formula: BCE = -[y*log(ŷ) + (1-y)*log(1-ŷ)]

    Properties:
    - Heavily penalizes confident wrong predictions
    - Works with sigmoid output

    Use: Binary classification (yes/no, spam/not spam)

    Args:
        y_true: 0 or 1 (or list/Matrix of 0s and 1s)
        y_pred: probability in (0, 1)
        epsilon: small constant for numerical stability

    IMPORTANT: Add epsilon to prevent log(0) = -∞
    Use: log(y_pred + epsilon) and log(1 - y_pred + epsilon)

    Examples:
        bce(1, 0.9) ≈ 0.105  (good prediction, low loss)
        bce(1, 0.1) ≈ 2.303  (bad prediction, high loss)
    """
    # TODO: Implement
    pass


def binary_cross_entropy_derivative(y_true, y_pred, epsilon=1e-15):
    """
    Derivative of BCE.

    Formula: dBCE/dŷ = (ŷ - y) / (ŷ * (1 - ŷ))

    When combined with sigmoid, this simplifies nicely!
    """
    # TODO: Implement
    pass


def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Categorical Cross-Entropy: for multi-class classification

    Formula: CCE = -Σ(y_true * log(y_pred))

    Properties:
    - Works with one-hot encoded labels
    - Works with softmax output
    - Output is a probability distribution

    Use: Multi-class classification (digit recognition!)

    Args:
        y_true: one-hot vector, e.g., [0, 0, 1, 0] for class 2
        y_pred: probability distribution, e.g., [0.1, 0.2, 0.6, 0.1]
        epsilon: prevents log(0)

    Examples:
        True label: [0, 1, 0]  (class 1)
        Prediction: [0.1, 0.8, 0.1]  → loss ≈ 0.22  (good)
        Prediction: [0.4, 0.1, 0.5]  → loss ≈ 2.30  (bad)
    """
    # TODO: Implement
    # Hint: Only non-zero y_true values contribute to the sum
    pass


def categorical_cross_entropy_derivative(y_true, y_pred, epsilon=1e-15):
    """
    Derivative of categorical cross-entropy.

    Formula: dCCE/dŷ = -y_true / y_pred

    When combined with softmax, simplifies to: ŷ - y_true
    """
    # TODO: Implement
    pass


# Utility function for later
def one_hot_encode(label, num_classes):
    """
    Convert integer label to one-hot vector.

    Args:
        label: integer in [0, num_classes)
        num_classes: total number of classes

    Returns:
        List with 1 at position 'label', 0 elsewhere

    Examples:
        one_hot_encode(2, 5) → [0, 0, 1, 0, 0]
        one_hot_encode(0, 3) → [1, 0, 0]
    """
    # TODO: Implement
    pass
```

---

## ✅ Test Your Implementations

Create `python/tests/test_activations.py`:

```python
"""Test activation functions"""

import sys
import os
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.activations import (
    sigmoid, sigmoid_derivative,
    relu, relu_derivative,
    tanh, tanh_derivative,
    softmax
)
from core.matrix import Matrix


def test_sigmoid():
    print("Testing sigmoid...", end=" ")

    # Test scalar
    assert abs(sigmoid(0) - 0.5) < 1e-6, "sigmoid(0) should be 0.5"
    assert sigmoid(100) > 0.99, "sigmoid(large) should be close to 1"
    assert sigmoid(-100) < 0.01, "sigmoid(-large) should be close to 0"

    # Test list
    result = sigmoid([0, 1, -1])
    assert abs(result[0] - 0.5) < 1e-6

    # Test Matrix
    m = Matrix([[0, 1], [-1, 2]])
    result = sigmoid(m)
    assert abs(result.data[0][0] - 0.5) < 1e-6

    print("✓ PASSED")


def test_sigmoid_derivative():
    print("Testing sigmoid derivative...", end=" ")

    # σ'(x) = σ(x) * (1 - σ(x))
    # At x=0: σ(0) = 0.5, so σ'(0) = 0.5 * 0.5 = 0.25
    sig_0 = sigmoid(0)
    deriv = sigmoid_derivative(sig_0)
    assert abs(deriv - 0.25) < 1e-6

    print("✓ PASSED")


def test_relu():
    print("Testing ReLU...", end=" ")

    assert relu(5) == 5
    assert relu(0) == 0
    assert relu(-5) == 0

    result = relu([-2, -1, 0, 1, 2])
    assert result == [0, 0, 0, 1, 2]

    print("✓ PASSED")


def test_relu_derivative():
    print("Testing ReLU derivative...", end=" ")

    assert relu_derivative(5) == 1
    assert relu_derivative(0) == 0
    assert relu_derivative(-5) == 0

    print("✓ PASSED")


def test_tanh():
    print("Testing tanh...", end=" ")

    assert abs(tanh(0)) < 1e-6, "tanh(0) should be 0"
    assert tanh(100) > 0.99, "tanh(large) should be close to 1"
    assert tanh(-100) < -0.99, "tanh(-large) should be close to -1"

    print("✓ PASSED")


def test_softmax():
    print("Testing softmax...", end=" ")

    # Test that outputs sum to 1
    result = softmax([1, 2, 3])
    total = sum(result)
    assert abs(total - 1.0) < 1e-6, "Softmax should sum to 1"

    # Test that larger inputs get higher probabilities
    assert result[2] > result[1] > result[0], "Larger inputs should have higher probability"

    # Test numerical stability with large numbers
    result = softmax([1000, 1001, 1002])
    assert not any(math.isinf(x) or math.isnan(x) for x in result), "Should handle large numbers"

    print("✓ PASSED")


def run_all_tests():
    print("\n" + "="*60)
    print(" 🧪 TESTING ACTIVATION FUNCTIONS")
    print("="*60 + "\n")

    tests = [
        test_sigmoid,
        test_sigmoid_derivative,
        test_relu,
        test_relu_derivative,
        test_tanh,
        test_softmax,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("🎉 ALL ACTIVATION TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
```

Create `python/tests/test_loss.py`:

```python
"""Test loss functions"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.loss import (
    mean_squared_error,
    mse_derivative,
    binary_cross_entropy,
    categorical_cross_entropy,
    one_hot_encode
)


def test_mse():
    print("Testing MSE...", end=" ")

    # Perfect prediction
    loss = mean_squared_error([1, 2, 3], [1, 2, 3])
    assert abs(loss) < 1e-6, "Perfect prediction should have 0 loss"

    # Known case
    loss = mean_squared_error([0, 0], [1, 1])
    assert abs(loss - 1.0) < 1e-6, "MSE([0,0], [1,1]) should be 1.0"

    print("✓ PASSED")


def test_binary_cross_entropy():
    print("Testing Binary Cross-Entropy...", end=" ")

    # Good prediction (true=1, pred=0.9)
    loss = binary_cross_entropy(1, 0.9)
    assert loss < 0.2, "Good prediction should have low loss"

    # Bad prediction (true=1, pred=0.1)
    loss = binary_cross_entropy(1, 0.1)
    assert loss > 2.0, "Bad prediction should have high loss"

    print("✓ PASSED")


def test_categorical_cross_entropy():
    print("Testing Categorical Cross-Entropy...", end=" ")

    # True label: class 1
    y_true = [0, 1, 0]

    # Good prediction
    y_pred = [0.1, 0.8, 0.1]
    loss = categorical_cross_entropy(y_true, y_pred)
    assert loss < 0.3, "Good prediction should have low loss"

    # Bad prediction
    y_pred = [0.4, 0.1, 0.5]
    loss = categorical_cross_entropy(y_true, y_pred)
    assert loss > 2.0, "Bad prediction should have high loss"

    print("✓ PASSED")


def test_one_hot_encode():
    print("Testing one-hot encoding...", end=" ")

    assert one_hot_encode(0, 3) == [1, 0, 0]
    assert one_hot_encode(2, 5) == [0, 0, 1, 0, 0]
    assert sum(one_hot_encode(3, 10)) == 1

    print("✓ PASSED")


def run_all_tests():
    print("\n" + "="*60)
    print(" 🧪 TESTING LOSS FUNCTIONS")
    print("="*60 + "\n")

    tests = [
        test_mse,
        test_binary_cross_entropy,
        test_categorical_cross_entropy,
        test_one_hot_encode,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("🎉 ALL LOSS TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
```

---

## 🎓 Understanding Questions

Before moving on, make sure you understand:

### 1. Why Non-Linearity?
What happens if you stack multiple linear layers without activation functions?

**Hint**: (AB)C = A(BC) for matrices. Multiple linear layers collapse to one!

### 2. Vanishing Gradients
Why do sigmoid and tanh cause vanishing gradients for large |x|?

**Hint**: Plot their derivatives. What happens when |x| > 5?

### 3. Why ReLU is Popular
What advantages does ReLU have over sigmoid?

### 4. Softmax Properties
Why does softmax(-max(x)) give the same result as softmax(x)?

**Hint**: Try it! Both numerator and denominator get multiplied by the same constant.

### 5. Loss Function Choice
Why use cross-entropy for classification instead of MSE?

**Hint**: Cross-entropy has better gradient properties for probability outputs.

---

## 📚 Resources for This Week

### Watch:
- 3Blue1Brown: Neural Networks Chapter 2 (Gradient Descent)
- Khan Academy: Derivatives review

### Read:
- Activation functions: https://cs231n.github.io/neural-networks-1/#actfun
- Loss functions: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html

### Visualize:
- Play with activation functions: https://playground.tensorflow.org/

---

## 🚀 Getting Started

1. Create `python/core/activations.py`
2. Implement sigmoid and relu first (easiest)
3. Test as you go!
4. Then tackle softmax (trickiest due to numerical stability)
5. Create `python/core/loss.py`
6. Implement MSE first, then cross-entropy

**Estimated time**: 4-6 hours

---

## 💡 Hints

### For Sigmoid
```python
def sigmoid(x):
    if isinstance(x, Matrix):
        return x.apply(sigmoid)  # Recursive!
    elif isinstance(x, list):
        return [sigmoid(xi) for xi in x]
    else:
        return 1 / (1 + math.exp(-x))
```

### For Softmax Numerical Stability
```python
# Bad: can overflow
exp_x = [math.exp(xi) for xi in x]

# Good: subtract max first
max_x = max(x)
exp_x = [math.exp(xi - max_x) for xi in x]
```

### For Loss Functions
```python
# Always flatten Matrix to list for loss computation
def _flatten(x):
    if isinstance(x, Matrix):
        return [item for row in x.data for item in row]
    return x
```

---

## ✅ Success Criteria

- All tests pass for activations
- All tests pass for loss functions
- You understand why each function is used
- You can explain the numerical stability trick in softmax

---

**Ready? Start with activations.py! Show me your code when you're done or if you get stuck!** 🚀
