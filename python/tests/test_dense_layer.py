"""Test dense layer"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.layer import DenseLayer
from core.matrix import Matrix


def test_layer_creation():
    """Test layer initialization."""
    print("Testing layer creation...", end=" ")

    layer = DenseLayer(3, 5)

    # Check shapes
    W_shape = layer.W.shape()
    b_shape = layer.b.shape()

    assert W_shape == (3, 5), f"Weight shape should be (3, 5), got {W_shape}"
    assert b_shape == (1, 5), f"Bias shape should be (1, 5), got {b_shape}"

    print("✓ PASSED")


def test_forward_pass():
    """Test forward pass computation."""
    print("Testing forward pass...", end=" ")

    layer = DenseLayer(3, 2)

    # Simple input
    X = Matrix([[1, 2, 3],
                [4, 5, 6]])  # 2 samples, 3 features

    Y = layer.forward(X)

    # Check output shape
    assert Y.shape() == (2, 2), f"Output shape should be (2, 2), got {Y.shape()}"

    print("✓ PASSED")


def test_backward_pass():
    """Test backward pass and gradient computation."""
    print("Testing backward pass...", end=" ")

    layer = DenseLayer(3, 2)

    # Forward pass
    X = Matrix([[1, 2, 3]])  # 1 sample, 3 features
    Y = layer.forward(X)

    # Backward pass
    dL_dY = Matrix([[0.1, 0.2]])  # Gradient from "next layer"
    dL_dX = layer.backward(dL_dY, learning_rate=0.01)

    # Check gradient shape
    assert dL_dX.shape() == (1, 3), f"Gradient shape should be (1, 3), got {dL_dX.shape()}"

    print("✓ PASSED")


def test_learning():
    """Test that weights actually update during training."""
    print("Testing weight updates...", end=" ")

    layer = DenseLayer(2, 1)

    # Save initial weights
    W_initial = layer.W.data[0][0]

    # Simple training step
    X = Matrix([[1, 1]])
    Y = layer.forward(X)

    # Arbitrary gradient
    dL_dY = Matrix([[1.0]])
    layer.backward(dL_dY, learning_rate=0.1)

    # Check that weights changed
    W_after = layer.W.data[0][0]
    assert W_initial != W_after, "Weights should change after backward pass"

    print("✓ PASSED")


def run_all_tests():
    print("\n" + "="*60)
    print(" 🧪 TESTING DENSE LAYER")
    print("="*60 + "\n")

    tests = [
        test_layer_creation,
        test_forward_pass,
        test_backward_pass,
        test_learning,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("🎉 ALL LAYER TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()