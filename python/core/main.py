
import sys
import time
sys.path.insert(0, '..')

from matrix import Matrix, zeros, ones, identity

def test():
    matrix = Matrix([[1, 2.7, 3.9], [4, 5, 6]])
    print("Original Matrix:")
    print(matrix)
    print("Display function:")
    matrix.__repr__()
    matrix.T().__repr__()

    matrix2 = Matrix([[1, 2.2, 3], [4.1, 5, 6]])
    matrix2.__repr__()
    print("Matrix equality test (should be True):", matrix == matrix2)
    print("Matrix sum:")
    sum_matrix = matrix.__add__(matrix2)
    sum_matrix.__repr__()
    print("Matrix difference:")
    diff_matrix = matrix.__sub__(matrix2)
    diff_matrix.__repr__()
    print("Matrix wise multiplication:")
    wise_mult_matrix = matrix * matrix2
    string = wise_mult_matrix.__repr__()
    print(string)
    def square_root(x):
        return x ** 0.5

    print("Applying square root function to each element:")
    applied_matrix = matrix.apply(square_root)
    applied_matrix.__repr__()

    matrix3 = Matrix([[1, 2], [3, 4], [5, 6]])
    print("Matrix multiplication result:")
    mult_matrix = matrix.dot(matrix3)
    mult_matrix.__repr__()

    # 4 x 4 Matrix tests
    matrix4 = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    identity(4).__repr__()
    zeros(4, 4).__repr__()
    ones(4, 4).__repr__()
    mult_4x4 = matrix4.dot(identity(4))
    mult_4x4.__repr__()
    mult_4x4 = matrix4.__mul__(identity(4))
    mult_4x4.__repr__()




def benchmark_matmul(n):
    """Benchmark matrix multiplication for nxn matrices."""
    print(f"\nBenchmarking {n}×{n} matrix multiplication...")
    
    # Create two random-ish matrices
    a = Matrix([[i + j for j in range(n)] for i in range(n)])
    b = Matrix([[i * j + 1 for j in range(n)] for i in range(n)])
    
    start = time.time()
    c = a.dot(b)
    end = time.time()
    
    elapsed = end - start
    print(f"Time: {elapsed:.4f} seconds")
    print(f"Operations: {n**3:,} multiplications")
    print(f"Speed: {n**3 / elapsed / 1e6:.2f} million ops/second")
    
    return elapsed

if __name__ == "__main__":
    sizes = [100, 500, 1000, 2000]
    
    print("="*60)
    print("MATRIX MULTIPLICATION BENCHMARK")
    print("="*60)
    
    for n in sizes:
        benchmark_matmul(n)
        print()
    
    print("="*60)
    print("\nNotice how time scales with O(n³):")
    print("Doubling size → ~8x slower (2³ = 8)")
    print("="*60)

