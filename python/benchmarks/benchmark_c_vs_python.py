import sys
import time
sys.path.insert(0, '..')

from core.matrix import Matrix as PyMatrix
from bindings.matrix_ctypes import Matrix as CMatrix

def benchmark_comparison():
    sizes = [100, 500, 1000]

    print("\n" + "="*70)
    print(" ⚡ C vs Pure Python Performance")
    print("="*70)
    print(f"{'Size':<10} {'Pure Python':<15} {'C Library':<15} {'Speedup':<10}")
    print("-"*70)

    for n in sizes:
        # Create test matrices
        data = [[float(i+j) for j in range(n)] for i in range(n)]

        # Pure Python
        start = time.time()
        a = PyMatrix(data)
        b = PyMatrix(data)
        c = a.dot(b)
        py_time = time.time() - start

        # C library
        start = time.time()
        a = CMatrix(data)
        b = CMatrix(data)
        c = a.dot(b)
        c_time = time.time() - start

        speedup = py_time / c_time

        print(f"{n}x{n:<6} {py_time:>12.4f}s {c_time:>12.6f}s {speedup:>8.1f}x")

    print("="*70)

if __name__ == "__main__":
    benchmark_comparison()