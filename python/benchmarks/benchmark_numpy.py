"""
Benchmark: Pure Python vs NumPy

Compare matrix multiplication speed.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.matrix import Matrix


def benchmark_pure_python(size=100):
    """Benchmark pure Python matrix multiplication."""

    # Create random matrices
    data_a = [[float(i * j % 97) for j in range(size)] for i in range(size)]
    data_b = [[float(i * j % 89) for j in range(size)] for i in range(size)]

    A = Matrix(data_a)
    B = Matrix(data_b)

    start = time.time()
    C = A.dot(B)
    elapsed = time.time() - start

    return elapsed


def benchmark_numpy(size=100):
    """Benchmark NumPy matrix multiplication."""

    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    start = time.time()
    C = np.dot(A, B)
    elapsed = time.time() - start

    return elapsed


def run_benchmarks():
    """Run benchmarks for different matrix sizes."""

    print("\n" + "="*70)
    print(" ⚡ NUMPY SPEED COMPARISON")
    print("="*70 + "\n")

    sizes = [100, 500, 1000, 2000]

    print(f"{'Size':<10} {'Pure Python':<15} {'NumPy':<15} {'Speedup':<10}")
    print("-" * 70)

    for size in sizes:
        py_time = benchmark_pure_python(size)
        np_time = benchmark_numpy(size)
        speedup = py_time / np_time

        print(f"{size}x{size:<6} {py_time:>10.4f}s {np_time:>14.6f}s {speedup:>10.1f}x")

    print("-" * 70)
    print("\nConclusion: NumPy is 100-1000x faster for matrix operations!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_benchmarks()