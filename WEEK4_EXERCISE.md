# Phase 4: CUDA GPU Acceleration

**Goal**: Implement neural network operations on your RTX 3080 GPU using CUDA, achieving 20-50x speedup over optimized C code.

This is where things get **really exciting**! You'll learn GPU programming and see massive performance gains.

---

## 🎯 What You'll Learn

1. **GPU Architecture**: How GPUs differ from CPUs
2. **CUDA Programming**: Kernels, threads, blocks, grids
3. **Memory Hierarchy**: Global, shared, register memory
4. **Optimization**: Coalesced access, shared memory, occupancy
5. **Profiling**: nvprof, Nsight Compute
6. **Python Integration**: Calling CUDA from Python

---

## 📋 Prerequisites

### Software Setup (Windows)

1. **CUDA Toolkit** (NVIDIA required)
   ```
   Download from: https://developer.nvidia.com/cuda-downloads
   Version: CUDA 12.x recommended
   ```

2. **Visual Studio 2022** (for nvcc compiler)
   - Install "Desktop development with C++"
   - Include Windows SDK

3. **Verify Installation**
   ```bash
   nvcc --version
   nvidia-smi  # Check GPU status
   ```

4. **Python CUDA Libraries**
   ```bash
   pip install pycuda
   # Or for simpler integration:
   pip install cupy
   ```

---

## 🚀 Exercise 4.1: Hello CUDA

### Goal
Write your first CUDA kernel and understand the GPU execution model.

### `cuda/hello.cu`
```cuda
#include <stdio.h>
#include <cuda_runtime.h>

/**
 * Your first CUDA kernel!
 *
 * __global__ means this runs on the GPU
 * It's called by CPU, executed on GPU
 */
__global__ void hello_kernel() {
    // Every thread has unique IDs
    int thread_id = threadIdx.x;           // Thread within block
    int block_id = blockIdx.x;             // Block within grid
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Hello from block %d, thread %d (global ID %d)\\n",
           block_id, thread_id, global_id);
}

int main() {
    printf("Launching kernel with 2 blocks, 4 threads each...\\n\\n");

    // Launch kernel: <<<num_blocks, threads_per_block>>>
    hello_kernel<<<2, 4>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\\n", cudaGetErrorString(err));
        return 1;
    }

    printf("\\nKernel completed successfully!\\n");
    return 0;
}
```

### Compile and Run
```bash
nvcc -o hello hello.cu
./hello

# Output:
# Launching kernel with 2 blocks, 4 threads each...
#
# Hello from block 0, thread 0 (global ID 0)
# Hello from block 0, thread 1 (global ID 1)
# Hello from block 0, thread 2 (global ID 2)
# Hello from block 0, thread 3 (global ID 3)
# Hello from block 1, thread 0 (global ID 4)
# Hello from block 1, thread 1 (global ID 5)
# Hello from block 1, thread 2 (global ID 6)
# Hello from block 1, thread 3 (global ID 7)
```

### Understanding the Execution Model

```
Grid (entire computation)
│
├─ Block 0 (group of threads)
│  ├─ Thread 0
│  ├─ Thread 1
│  ├─ Thread 2
│  └─ Thread 3
│
└─ Block 1
   ├─ Thread 0
   ├─ Thread 1
   ├─ Thread 2
   └─ Thread 3

RTX 3080: 8704 CUDA cores!
You can launch MILLIONS of threads simultaneously!
```

---

## 🚀 Exercise 4.2: Vector Addition

### Goal
Perform element-wise vector addition on GPU and understand memory transfer.

### `cuda/vector_add.cu`
```cuda
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/**
 * Vector addition kernel
 * Each thread computes one element: c[i] = a[i] + b[i]
 */
__global__ void vector_add_kernel(
    const float *a,
    const float *b,
    float *c,
    int n
) {
    // Calculate global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check (important!)
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * Wrapper function for vector addition
 */
void vector_add_gpu(
    const float *h_a,   // h_ prefix = host (CPU) memory
    const float *h_b,
    float *h_c,
    int n
) {
    // Device (GPU) pointers
    float *d_a, *d_b, *d_c;  // d_ prefix = device memory

    size_t bytes = n * sizeof(float);

    // 1. Allocate device memory
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // 2. Copy input data from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 3. Launch kernel
    int threads_per_block = 256;  // Typical value
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    printf("Launching %d blocks with %d threads each\\n",
           num_blocks, threads_per_block);

    vector_add_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\\n", cudaGetErrorString(err));
        return;
    }

    // 4. Copy result from device to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // 5. Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

/**
 * CPU version for comparison
 */
void vector_add_cpu(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 10000000;  // 10 million elements

    // Allocate host memory
    float *h_a = (float*)malloc(N * sizeof(float));
    float *h_b = (float*)malloc(N * sizeof(float));
    float *h_c_gpu = (float*)malloc(N * sizeof(float));
    float *h_c_cpu = (float*)malloc(N * sizeof(float));

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // GPU computation with timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vector_add_gpu(h_a, h_b, h_c_gpu, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // CPU computation with timing
    clock_t cpu_start = clock();
    vector_add_cpu(h_a, h_b, h_c_cpu, N);
    clock_t cpu_end = clock();
    float cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Verify results match
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_c_gpu[i] != h_c_cpu[i]) {
            errors++;
            if (errors < 5) {
                printf("Mismatch at %d: GPU=%f, CPU=%f\\n",
                       i, h_c_gpu[i], h_c_cpu[i]);
            }
        }
    }

    printf("\\n");
    printf("Vector size: %d elements\\n", N);
    printf("CPU time: %.3f ms\\n", cpu_time);
    printf("GPU time: %.3f ms\\n", gpu_time);
    printf("Speedup: %.2fx\\n", cpu_time / gpu_time);
    printf("Errors: %d / %d\\n", errors, N);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_a);
    free(h_b);
    free(h_c_gpu);
    free(h_c_cpu);

    return 0;
}
```

### Compile and Run
```bash
nvcc -O3 -o vector_add vector_add.cu
./vector_add

# Expected output on RTX 3080:
# Vector size: 10000000 elements
# CPU time: ~30-50 ms
# GPU time: ~2-5 ms
# Speedup: 10-15x
```

---

## 🚀 Exercise 4.3: Matrix Multiplication - Naive

### Goal
Implement matrix multiplication on GPU (basic version).

### `cuda/matmul_naive.cu`
```cuda
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/**
 * Naive matrix multiplication kernel
 * Each thread computes ONE element of the result matrix
 *
 * C[row, col] = sum(A[row, k] * B[k, col]) for all k
 */
__global__ void matmul_kernel(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K
) {
    // Calculate which element this thread computes
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (row < M && col < K) {
        float sum = 0.0f;

        // Compute dot product of row from A with column from B
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }

        C[row * K + col] = sum;
    }
}

/**
 * Host function to launch matrix multiplication
 */
void matmul_gpu(
    const float *h_A,
    const float *h_B,
    float *h_C,
    int M, int N, int K
) {
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    // Launch configuration
    dim3 threads_per_block(16, 16);  // 16x16 = 256 threads per block
    dim3 num_blocks(
        (K + threads_per_block.x - 1) / threads_per_block.x,
        (M + threads_per_block.y - 1) / threads_per_block.y
    );

    // Launch kernel
    matmul_kernel<<<num_blocks, threads_per_block>>>(d_A, d_B, d_C, M, N, K);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel error: %s\\n", cudaGetErrorString(err));
    }

    // Copy result back
    cudaMemcpy(h_C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/**
 * CPU matrix multiplication for verification
 */
void matmul_cpu(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

int main() {
    const int M = 1024, N = 1024, K = 1024;

    // Allocate host memory
    float *h_A = (float*)malloc(M * N * sizeof(float));
    float *h_B = (float*)malloc(N * K * sizeof(float));
    float *h_C_gpu = (float*)malloc(M * K * sizeof(float));
    float *h_C_cpu = (float*)malloc(M * K * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < M * N; i++) h_A[i] = (float)(rand() % 100) / 10.0f;
    for (int i = 0; i < N * K; i++) h_B[i] = (float)(rand() % 100) / 10.0f;

    printf("Matrix dimensions: (%d×%d) × (%d×%d) = (%d×%d)\\n\\n",
           M, N, N, K, M, K);

    // GPU computation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_gpu(h_A, h_B, h_C_gpu, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time;
    cudaEventElapsedTime(&gpu_time, start, stop);

    // CPU computation
    clock_t cpu_start = clock();
    matmul_cpu(h_A, h_B, h_C_cpu, M, N, K);
    clock_t cpu_end = clock();
    float cpu_time = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Verify correctness
    float max_error = 0.0f;
    for (int i = 0; i < M * K; i++) {
        float error = fabsf(h_C_gpu[i] - h_C_cpu[i]);
        if (error > max_error) max_error = error;
    }

    printf("CPU time: %.3f ms\\n", cpu_time);
    printf("GPU time: %.3f ms\\n", gpu_time);
    printf("Speedup: %.2fx\\n", cpu_time / gpu_time);
    printf("Max error: %e\\n", max_error);
    printf("Performance: %.2f GFLOPS\\n",
           2.0 * M * N * K / (gpu_time * 1e6));

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);

    return 0;
}
```

### Expected Performance (RTX 3080)
```
Matrix dimensions: (1024×1024) × (1024×1024) = (1024×1024)

CPU time: ~4000 ms
GPU time: ~15-20 ms
Speedup: 200-250x
Performance: ~100 GFLOPS
```

But we can do MUCH better! The RTX 3080 can do **30 TFLOPS** (30,000 GFLOPS)!

---

## 🚀 Exercise 4.4: Matrix Multiplication - Optimized with Shared Memory

### Goal
Use shared memory to achieve 10-20x additional speedup!

### `cuda/matmul_shared.cu`
```cuda
#define TILE_SIZE 16

/**
 * Optimized matrix multiplication using shared memory tiling
 *
 * Key optimization: Load data into fast shared memory
 * instead of reading from slow global memory repeatedly
 */
__global__ void matmul_tiled_kernel(
    const float *A,
    const float *B,
    float *C,
    int M, int N, int K
) {
    // Shared memory for tiles (fast, on-chip memory!)
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Load tile from A into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < N) {
            A_tile[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B into shared memory
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < N && col < K) {
            B_tile[threadIdx.y][threadIdx.x] = B[b_row * K + col];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure all threads loaded their data
        __syncthreads();

        // Compute partial dot product using shared memory
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result
    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}
```

### Why This Is Faster

```
Global Memory Access: ~400 cycles latency
Shared Memory Access: ~4 cycles latency

100x faster memory access!

For 1024×1024 matrix:
- Naive: Each element reads 1024 values from global memory
- Tiled: Each element reads mostly from shared memory
- Result: 10-20x speedup!
```

### Expected Performance
```
Matrix dimensions: (1024×1024) × (1024×1024)

Naive GPU: ~15-20 ms (~100 GFLOPS)
Tiled GPU: ~1-2 ms (~1000 GFLOPS)

Speedup: 10-15x over naive
Performance: 10x better utilization!
```

---

## 🔗 Exercise 4.5: Python Integration with PyCUDA

### Install PyCUDA
```bash
pip install pycuda
```

### `python/cuda/matmul_cuda.py`
```python
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np
import time

# CUDA kernel code as a string
cuda_code = """
__global__ void matmul_kernel(
    float *A, float *B, float *C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}
"""

# Compile CUDA code
mod = SourceModule(cuda_code)
matmul_kernel = mod.get_function("matmul_kernel")

def matmul_cuda(A, B):
    """
    Matrix multiplication using CUDA from Python!

    Args:
        A: NumPy array (M×N)
        B: NumPy array (N×K)

    Returns:
        C: NumPy array (M×K)
    """
    M, N = A.shape
    N2, K = B.shape
    assert N == N2, "Inner dimensions must match"

    # Allocate result
    C = np.zeros((M, K), dtype=np.float32)

    # Convert to contiguous arrays (required for GPU)
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    # Launch configuration
    block_size = (16, 16, 1)
    grid_size = (
        (K + block_size[0] - 1) // block_size[0],
        (M + block_size[1] - 1) // block_size[1],
        1
    )

    # Call CUDA kernel
    matmul_kernel(
        drv.In(A), drv.In(B), drv.Out(C),
        np.int32(M), np.int32(N), np.int32(K),
        block=block_size,
        grid=grid_size
    )

    return C

# Test it!
if __name__ == "__main__":
    n = 1024

    print(f"Testing {n}×{n} matrix multiplication...\\n")

    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)

    # NumPy (CPU)
    start = time.time()
    C_numpy = A @ B
    numpy_time = time.time() - start

    # CUDA (GPU)
    start = time.time()
    C_cuda = matmul_cuda(A, B)
    cuda_time = time.time() - start

    # Verify correctness
    error = np.max(np.abs(C_numpy - C_cuda))

    print(f"NumPy time: {numpy_time*1000:.3f} ms")
    print(f"CUDA time: {cuda_time*1000:.3f} ms")
    print(f"Speedup: {numpy_time/cuda_time:.2f}x")
    print(f"Max error: {error:.6e}")
```

---

## 🧠 Exercise 4.6: Neural Network Forward Pass on GPU

### Goal
Implement a complete forward pass on GPU!

### `cuda/nn_forward.cu`
```cuda
/**
 * Dense layer forward pass on GPU
 * Y = sigmoid(X @ W + b)
 */

__global__ void sigmoid_kernel(float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

__global__ void add_bias_kernel(
    float *output,
    const float *bias,
    int batch_size,
    int output_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < output_size) {
        output[row * output_size + col] += bias[col];
    }
}

// Use matmul_tiled_kernel from Exercise 4.4 for matrix multiplication

void forward_pass_gpu(
    const float *X,          // Input: (batch_size, input_size)
    const float *W,          // Weights: (input_size, output_size)
    const float *b,          // Bias: (output_size,)
    float *Y,                // Output: (batch_size, output_size)
    int batch_size,
    int input_size,
    int output_size
) {
    // 1. Matrix multiply: temp = X @ W
    matmul_tiled_kernel<<<...>>>(X, W, Y, batch_size, input_size, output_size);

    // 2. Add bias: Y = temp + b
    dim3 threads(16, 16);
    dim3 blocks(
        (output_size + 15) / 16,
        (batch_size + 15) / 16
    );
    add_bias_kernel<<<blocks, threads>>>(Y, b, batch_size, output_size);

    // 3. Apply activation: Y = sigmoid(Y)
    int total_elements = batch_size * output_size;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    sigmoid_kernel<<<num_blocks, threads_per_block>>>(Y, total_elements);
}
```

---

## ✅ Success Criteria

- [ ] Hello CUDA runs and prints thread IDs
- [ ] Vector addition shows 10-15x speedup
- [ ] Naive matmul shows 100-200x speedup over CPU
- [ ] Tiled matmul shows 10-20x speedup over naive
- [ ] PyCUDA integration works from Python
- [ ] Forward pass runs on GPU
- [ ] Results match CPU within 1e-5
- [ ] You understand shared memory optimization

---

## 📚 Key Concepts

### Memory Hierarchy
```
Registers:     <1 cycle,    64KB per SM
Shared Memory: 4 cycles,    100KB per SM
L1 Cache:      10 cycles,   128KB per SM
L2 Cache:      100 cycles,  6MB total
Global Memory: 400 cycles,  10GB total (RTX 3080)
```

### Optimization Checklist
- [ ] Coalesced memory access (consecutive threads access consecutive memory)
- [ ] Shared memory usage (load once, use many times)
- [ ] Minimize divergence (avoid if/else in warps)
- [ ] High occupancy (keep all SMs busy)
- [ ] Minimize host-device transfers

---

## 🎓 Understanding Questions

1. **Why use threads instead of cores?**
   - GPUs hide latency with massive parallelism
   - While some threads wait for memory, others compute

2. **What is a warp?**
   - 32 threads execute together in lockstep
   - SIMT: Single Instruction Multiple Thread

3. **Why shared memory?**
   - 100x faster than global memory
   - Data reuse within a block

4. **When is GPU faster?**
   - Large data (amortize transfer cost)
   - Parallel computation (matrix operations)
   - Memory-bound operations benefit most

---

## 🚀 Next: Complete Neural Network Training on GPU

Once you complete Phase 4, you'll have:
- Matrix operations on GPU
- Activation functions on GPU
- Forward pass working

Phase 5 will add:
- Backward pass (backpropagation)
- Weight updates (SGD optimizer)
- Complete training loop
- **Goal**: Train MNIST classifier entirely on GPU in <30 seconds!

---

**Start with Exercise 4.1 (Hello CUDA)! Get familiar with the GPU execution model first.**

Show me your code as you progress!
