# Phase 5: Complete Neural Network Training on GPU

**Goal**: Implement a fully functional neural network with forward pass, backward pass, and training - all running on your RTX 3080 GPU!

**End Result**: Train MNIST digit classifier to >97% accuracy in <30 seconds on GPU.

---

## 🎯 What You'll Build

By the end of this phase, you'll have:

1. **Complete forward pass on GPU** (all layers, all activations)
2. **Complete backward pass on GPU** (backpropagation with gradients)
3. **GPU-accelerated optimizers** (SGD, momentum, Adam)
4. **Full training pipeline** (data loading, batching, metrics)
5. **Python interface** (clean API for using your GPU library)
6. **MNIST classifier** achieving state-of-the-art performance

---

## 📋 Phase Overview

```
Phase 5: Complete Neural Network (Weeks 12-16)
├── Exercise 5.1: Activation Functions on GPU
├── Exercise 5.2: Loss Functions on GPU
├── Exercise 5.3: Backward Pass (Backpropagation) on GPU
├── Exercise 5.4: Optimizers on GPU (SGD, Momentum, Adam)
├── Exercise 5.5: Complete Training Loop
├── Exercise 5.6: Python Integration (Clean API)
└── Final Project: MNIST Digit Classifier
```

---

## 🚀 Exercise 5.1: Activation Functions on GPU

### Goal
Implement all activation functions and their derivatives as CUDA kernels.

### `cuda/activations.cu`

```cuda
#ifndef ACTIVATIONS_CU
#define ACTIVATIONS_CU

#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// SIGMOID
// ============================================================================

/**
 * Sigmoid activation: f(x) = 1 / (1 + e^(-x))
 * Range: (0, 1)
 */
__global__ void sigmoid_forward_kernel(
    const float *input,
    float *output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

/**
 * Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
 * Note: Expects sigmoid OUTPUT as input
 */
__global__ void sigmoid_backward_kernel(
    const float *grad_output,    // Gradient from next layer
    const float *sigmoid_output, // Output of forward pass
    float *grad_input,           // Gradient to propagate
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sig = sigmoid_output[idx];
        grad_input[idx] = grad_output[idx] * sig * (1.0f - sig);
    }
}

// ============================================================================
// ReLU
// ============================================================================

/**
 * ReLU activation: f(x) = max(0, x)
 * Most popular for hidden layers
 */
__global__ void relu_forward_kernel(
    const float *input,
    float *output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

/**
 * ReLU derivative: f'(x) = 1 if x > 0 else 0
 * Note: Expects original INPUT (not output)
 */
__global__ void relu_backward_kernel(
    const float *grad_output,
    const float *input,
    float *grad_input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

// ============================================================================
// SOFTMAX
// ============================================================================

/**
 * Softmax activation: f(x_i) = e^x_i / sum(e^x_j)
 * Applied row-wise for batched inputs
 *
 * Each block handles one row (one sample in batch)
 * Uses shared memory for reduction
 */
__global__ void softmax_forward_kernel(
    const float *input,
    float *output,
    int batch_size,
    int num_classes
) {
    extern __shared__ float shared_mem[];
    float *shared_max = shared_mem;
    float *shared_sum = shared_mem + blockDim.x;

    int row = blockIdx.x;  // Which sample in batch
    if (row >= batch_size) return;

    int tid = threadIdx.x;
    const float *row_input = input + row * num_classes;
    float *row_output = output + row * num_classes;

    // Step 1: Find maximum value in row (for numerical stability)
    float local_max = -INFINITY;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_max = fmaxf(local_max, row_input[i]);
    }
    shared_max[tid] = local_max;
    __syncthreads();

    // Reduce to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_max[0];
    __syncthreads();

    // Step 2: Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float exp_val = expf(row_input[i] - max_val);
        row_output[i] = exp_val;
        local_sum += exp_val;
    }
    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduce to find sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    float sum = shared_sum[0];
    __syncthreads();

    // Step 3: Normalize
    for (int i = tid; i < num_classes; i += blockDim.x) {
        row_output[i] /= sum;
    }
}

/**
 * Softmax backward combined with cross-entropy
 * Gradient simplifies to: pred - target
 */
__global__ void softmax_crossentropy_backward_kernel(
    const float *predictions,  // Softmax output
    const float *targets,      // One-hot encoded
    float *grad_input,
    int batch_size,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_classes;

    if (idx < total) {
        // Simple: gradient = (prediction - target) / batch_size
        grad_input[idx] = (predictions[idx] - targets[idx]) / batch_size;
    }
}

// ============================================================================
// TANH
// ============================================================================

__global__ void tanh_forward_kernel(
    const float *input,
    float *output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tanhf(input[idx]);
    }
}

__global__ void tanh_backward_kernel(
    const float *grad_output,
    const float *tanh_output,
    float *grad_input,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float t = tanh_output[idx];
        grad_input[idx] = grad_output[idx] * (1.0f - t * t);
    }
}

// ============================================================================
// HOST WRAPPERS
// ============================================================================

void sigmoid_forward_gpu(const float *d_input, float *d_output, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sigmoid_forward_kernel<<<blocks, threads>>>(d_input, d_output, n);
}

void sigmoid_backward_gpu(
    const float *d_grad_output,
    const float *d_sigmoid_output,
    float *d_grad_input,
    int n
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sigmoid_backward_kernel<<<blocks, threads>>>(
        d_grad_output, d_sigmoid_output, d_grad_input, n
    );
}

void relu_forward_gpu(const float *d_input, float *d_output, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_forward_kernel<<<blocks, threads>>>(d_input, d_output, n);
}

void relu_backward_gpu(
    const float *d_grad_output,
    const float *d_input,
    float *d_grad_input,
    int n
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_backward_kernel<<<blocks, threads>>>(
        d_grad_output, d_input, d_grad_input, n
    );
}

void softmax_forward_gpu(
    const float *d_input,
    float *d_output,
    int batch_size,
    int num_classes
) {
    int threads = 256;
    int blocks = batch_size;
    int shared_mem_size = 2 * threads * sizeof(float);

    softmax_forward_kernel<<<blocks, threads, shared_mem_size>>>(
        d_input, d_output, batch_size, num_classes
    );
}

void softmax_crossentropy_backward_gpu(
    const float *d_predictions,
    const float *d_targets,
    float *d_grad_input,
    int batch_size,
    int num_classes
) {
    int total = batch_size * num_classes;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    softmax_crossentropy_backward_kernel<<<blocks, threads>>>(
        d_predictions, d_targets, d_grad_input, batch_size, num_classes
    );
}

#endif // ACTIVATIONS_CU
```

---

## 🚀 Exercise 5.2: Loss Functions on GPU

### `cuda/loss.cu`

```cuda
#ifndef LOSS_CU
#define LOSS_CU

#include <cuda_runtime.h>

// ============================================================================
// CROSS-ENTROPY LOSS
// ============================================================================

/**
 * Categorical cross-entropy loss: -sum(target * log(pred))
 * Uses atomic operations for reduction
 */
__global__ void cross_entropy_loss_kernel(
    const float *predictions,
    const float *targets,
    float *loss,
    int batch_size,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_classes;

    if (idx < total) {
        float pred = predictions[idx];
        float target = targets[idx];

        // Add epsilon for numerical stability
        float epsilon = 1e-10f;
        pred = fmaxf(pred, epsilon);
        pred = fminf(pred, 1.0f - epsilon);

        if (target > 0.0f) {  // Only non-zero targets contribute
            float local_loss = -target * logf(pred);
            atomicAdd(loss, local_loss / batch_size);
        }
    }
}

/**
 * Mean Squared Error: (1/n) * sum((pred - target)^2)
 */
__global__ void mse_loss_kernel(
    const float *predictions,
    const float *targets,
    float *loss,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float diff = predictions[idx] - targets[idx];
        float local_loss = diff * diff / n;
        atomicAdd(loss, local_loss);
    }
}

// ============================================================================
// ACCURACY COMPUTATION
// ============================================================================

/**
 * Compute accuracy for classification
 * Each thread handles one sample
 */
__global__ void accuracy_kernel(
    const float *predictions,  // Softmax output (batch_size × num_classes)
    const float *targets,      // One-hot encoded
    int *correct_count,
    int batch_size,
    int num_classes
) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (sample_idx < batch_size) {
        const float *pred_row = predictions + sample_idx * num_classes;
        const float *target_row = targets + sample_idx * num_classes;

        // Find predicted class (argmax)
        int pred_class = 0;
        float max_pred = pred_row[0];
        for (int i = 1; i < num_classes; i++) {
            if (pred_row[i] > max_pred) {
                max_pred = pred_row[i];
                pred_class = i;
            }
        }

        // Find true class
        int true_class = 0;
        for (int i = 0; i < num_classes; i++) {
            if (target_row[i] > 0.5f) {
                true_class = i;
                break;
            }
        }

        // Increment if correct
        if (pred_class == true_class) {
            atomicAdd(correct_count, 1);
        }
    }
}

// ============================================================================
// HOST WRAPPERS
// ============================================================================

float cross_entropy_loss_gpu(
    const float *d_predictions,
    const float *d_targets,
    int batch_size,
    int num_classes
) {
    // Allocate device memory for loss (single value)
    float *d_loss;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    // Launch kernel
    int total = batch_size * num_classes;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    cross_entropy_loss_kernel<<<blocks, threads>>>(
        d_predictions, d_targets, d_loss, batch_size, num_classes
    );

    // Copy result back
    float h_loss;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);

    return h_loss;
}

float accuracy_gpu(
    const float *d_predictions,
    const float *d_targets,
    int batch_size,
    int num_classes
) {
    // Allocate device memory for count
    int *d_correct_count;
    cudaMalloc(&d_correct_count, sizeof(int));
    cudaMemset(d_correct_count, 0, sizeof(int));

    // Launch kernel
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    accuracy_kernel<<<blocks, threads>>>(
        d_predictions, d_targets, d_correct_count, batch_size, num_classes
    );

    // Copy result back
    int h_correct_count;
    cudaMemcpy(&h_correct_count, d_correct_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_correct_count);

    return (float)h_correct_count / batch_size;
}

#endif // LOSS_CU
```

---

## 🚀 Exercise 5.3: Backward Pass (Backpropagation) on GPU

### `cuda/backward.cu`

```cuda
#ifndef BACKWARD_CU
#define BACKWARD_CU

#include <cuda_runtime.h>

/**
 * Dense layer backward pass
 *
 * Given:
 *   - grad_output: gradient from next layer (batch_size × output_size)
 *   - input: saved input from forward pass (batch_size × input_size)
 *   - weights: current weights (input_size × output_size)
 *
 * Compute:
 *   - grad_weights: W gradient (input_size × output_size)
 *   - grad_bias: b gradient (output_size,)
 *   - grad_input: gradient to previous layer (batch_size × input_size)
 */

// ============================================================================
// GRADIENT COMPUTATION
// ============================================================================

/**
 * Compute weight gradients: grad_W = X^T @ grad_output
 * Uses tiled matrix multiplication for efficiency
 */
__global__ void compute_weight_gradients_kernel(
    const float *input,       // (batch_size × input_size)
    const float *grad_output, // (batch_size × output_size)
    float *grad_weights,      // (input_size × output_size)
    int batch_size,
    int input_size,
    int output_size
) {
    // Each thread computes one element of grad_weights
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // input_size dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // output_size dimension

    if (row < input_size && col < output_size) {
        float sum = 0.0f;

        // grad_W[i,j] = sum over batch: input[b,i] * grad_output[b,j]
        for (int b = 0; b < batch_size; b++) {
            sum += input[b * input_size + row] * grad_output[b * output_size + col];
        }

        grad_weights[row * output_size + col] = sum;
    }
}

/**
 * Compute bias gradients: grad_b = sum(grad_output, axis=0)
 * Sum over batch dimension
 */
__global__ void compute_bias_gradients_kernel(
    const float *grad_output,  // (batch_size × output_size)
    float *grad_bias,          // (output_size,)
    int batch_size,
    int output_size
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < output_size) {
        float sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            sum += grad_output[b * output_size + col];
        }
        grad_bias[col] = sum;
    }
}

/**
 * Compute input gradients: grad_X = grad_output @ W^T
 * Uses tiled matrix multiplication
 */
__global__ void compute_input_gradients_kernel(
    const float *grad_output,  // (batch_size × output_size)
    const float *weights,      // (input_size × output_size)
    float *grad_input,         // (batch_size × input_size)
    int batch_size,
    int input_size,
    int output_size
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // batch dimension
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // input_size dimension

    if (row < batch_size && col < input_size) {
        float sum = 0.0f;

        // grad_X[b,i] = sum: grad_output[b,j] * weights[i,j]
        for (int j = 0; j < output_size; j++) {
            sum += grad_output[row * output_size + j] * weights[col * output_size + j];
        }

        grad_input[row * input_size + col] = sum;
    }
}

// ============================================================================
// OPTIMIZED VERSION WITH SHARED MEMORY
// ============================================================================

#define TILE_SIZE 16

/**
 * Optimized weight gradient computation using shared memory
 */
__global__ void compute_weight_gradients_tiled_kernel(
    const float *input,
    const float *grad_output,
    float *grad_weights,
    int batch_size,
    int input_size,
    int output_size
) {
    __shared__ float input_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float grad_tile[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    int num_tiles = (batch_size + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Load tiles
        int input_row = t * TILE_SIZE + threadIdx.x;
        int grad_row = t * TILE_SIZE + threadIdx.x;

        if (row < input_size && input_row < batch_size) {
            input_tile[threadIdx.y][threadIdx.x] = input[input_row * input_size + row];
        } else {
            input_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (grad_row < batch_size && col < output_size) {
            grad_tile[threadIdx.y][threadIdx.x] = grad_output[grad_row * output_size + col];
        } else {
            grad_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += input_tile[threadIdx.y][k] * grad_tile[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < input_size && col < output_size) {
        grad_weights[row * output_size + col] = sum;
    }
}

// ============================================================================
// HOST WRAPPERS
// ============================================================================

void dense_layer_backward_gpu(
    const float *d_input,
    const float *d_grad_output,
    const float *d_weights,
    float *d_grad_weights,
    float *d_grad_bias,
    float *d_grad_input,
    int batch_size,
    int input_size,
    int output_size,
    bool use_tiled
) {
    // Compute weight gradients
    if (use_tiled) {
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks(
            (output_size + TILE_SIZE - 1) / TILE_SIZE,
            (input_size + TILE_SIZE - 1) / TILE_SIZE
        );
        compute_weight_gradients_tiled_kernel<<<blocks, threads>>>(
            d_input, d_grad_output, d_grad_weights,
            batch_size, input_size, output_size
        );
    } else {
        dim3 threads(16, 16);
        dim3 blocks(
            (output_size + 15) / 16,
            (input_size + 15) / 16
        );
        compute_weight_gradients_kernel<<<blocks, threads>>>(
            d_input, d_grad_output, d_grad_weights,
            batch_size, input_size, output_size
        );
    }

    // Compute bias gradients
    int threads_bias = 256;
    int blocks_bias = (output_size + threads_bias - 1) / threads_bias;
    compute_bias_gradients_kernel<<<blocks_bias, threads_bias>>>(
        d_grad_output, d_grad_bias, batch_size, output_size
    );

    // Compute input gradients
    dim3 threads_input(16, 16);
    dim3 blocks_input(
        (input_size + 15) / 16,
        (batch_size + 15) / 16
    );
    compute_input_gradients_kernel<<<blocks_input, threads_input>>>(
        d_grad_output, d_weights, d_grad_input,
        batch_size, input_size, output_size
    );
}

#endif // BACKWARD_CU
```

---

## 🚀 Exercise 5.4: Optimizers on GPU

### `cuda/optimizer.cu`

```cuda
#ifndef OPTIMIZER_CU
#define OPTIMIZER_CU

#include <cuda_runtime.h>

// ============================================================================
// STOCHASTIC GRADIENT DESCENT (SGD)
// ============================================================================

/**
 * Basic SGD update: W = W - learning_rate * grad_W
 */
__global__ void sgd_update_kernel(
    float *weights,
    const float *gradients,
    float learning_rate,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

// ============================================================================
// SGD WITH MOMENTUM
// ============================================================================

/**
 * SGD with momentum:
 *   velocity = momentum * velocity - learning_rate * gradient
 *   weight = weight + velocity
 */
__global__ void sgd_momentum_update_kernel(
    float *weights,
    const float *gradients,
    float *velocity,
    float learning_rate,
    float momentum,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        velocity[idx] = momentum * velocity[idx] - learning_rate * gradients[idx];
        weights[idx] += velocity[idx];
    }
}

// ============================================================================
// ADAM OPTIMIZER
// ============================================================================

/**
 * Adam optimizer: Adaptive Moment Estimation
 *
 * m = beta1 * m + (1 - beta1) * gradient
 * v = beta2 * v + (1 - beta2) * gradient^2
 * m_hat = m / (1 - beta1^t)
 * v_hat = v / (1 - beta2^t)
 * weight = weight - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
 */
__global__ void adam_update_kernel(
    float *weights,
    const float *gradients,
    float *m,              // First moment estimate
    float *v,              // Second moment estimate
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    int timestep,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float grad = gradients[idx];

        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;

        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

        // Compute bias-corrected first moment estimate
        float m_hat = m[idx] / (1.0f - powf(beta1, timestep));

        // Compute bias-corrected second raw moment estimate
        float v_hat = v[idx] / (1.0f - powf(beta2, timestep));

        // Update weights
        weights[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// ============================================================================
// WEIGHT DECAY (L2 REGULARIZATION)
// ============================================================================

/**
 * Apply weight decay: W = W * (1 - weight_decay * learning_rate)
 */
__global__ void weight_decay_kernel(
    float *weights,
    float learning_rate,
    float weight_decay,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weights[idx] *= (1.0f - learning_rate * weight_decay);
    }
}

// ============================================================================
// GRADIENT CLIPPING
// ============================================================================

/**
 * Clip gradients by norm
 * First pass: compute global norm
 * Second pass: scale if necessary
 */
__global__ void gradient_norm_kernel(
    const float *gradients,
    float *partial_norms,
    int n
) {
    extern __shared__ float shared_norms[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Each thread computes local squared norm
    float local_norm = 0.0f;
    if (idx < n) {
        float grad = gradients[idx];
        local_norm = grad * grad;
    }
    shared_norms[tid] = local_norm;
    __syncthreads();

    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_norms[tid] += shared_norms[tid + stride];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        partial_norms[blockIdx.x] = shared_norms[0];
    }
}

__global__ void gradient_clip_kernel(
    float *gradients,
    float clip_value,
    float norm,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && norm > clip_value) {
        gradients[idx] *= clip_value / norm;
    }
}

// ============================================================================
// HOST WRAPPERS
// ============================================================================

void sgd_update_gpu(
    float *d_weights,
    const float *d_gradients,
    float learning_rate,
    int n
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(d_weights, d_gradients, learning_rate, n);
}

void sgd_momentum_update_gpu(
    float *d_weights,
    const float *d_gradients,
    float *d_velocity,
    float learning_rate,
    float momentum,
    int n
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sgd_momentum_update_kernel<<<blocks, threads>>>(
        d_weights, d_gradients, d_velocity, learning_rate, momentum, n
    );
}

void adam_update_gpu(
    float *d_weights,
    const float *d_gradients,
    float *d_m,
    float *d_v,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    int timestep,
    int n
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    adam_update_kernel<<<blocks, threads>>>(
        d_weights, d_gradients, d_m, d_v,
        learning_rate, beta1, beta2, epsilon, timestep, n
    );
}

#endif // OPTIMIZER_CU
```

---

## 🚀 Exercise 5.5: Complete Training Loop

### `cuda/nn_gpu.cu`

```cuda
#include "activations.cu"
#include "loss.cu"
#include "backward.cu"
#include "optimizer.cu"
#include "matmul_shared.cu"  // From Phase 4

/**
 * Complete neural network structure
 */
typedef struct {
    // Network architecture
    int input_size;
    int hidden_size;
    int output_size;
    int batch_size;

    // Device memory - Layer 1 (input → hidden)
    float *d_W1;           // Weights (input_size × hidden_size)
    float *d_b1;           // Bias (hidden_size,)
    float *d_z1;           // Pre-activation (batch_size × hidden_size)
    float *d_a1;           // Activation (batch_size × hidden_size)
    float *d_grad_W1;      // Weight gradients
    float *d_grad_b1;      // Bias gradients
    float *d_grad_z1;      // Pre-activation gradients

    // Device memory - Layer 2 (hidden → output)
    float *d_W2;
    float *d_b2;
    float *d_z2;
    float *d_a2;           // Final output (softmax)
    float *d_grad_W2;
    float *d_grad_b2;
    float *d_grad_z2;

    // Optimizer state (for Adam)
    float *d_m_W1, *d_v_W1;
    float *d_m_b1, *d_v_b1;
    float *d_m_W2, *d_v_W2;
    float *d_m_b2, *d_v_b2;
    int timestep;

    // Input/output
    float *d_input;        // Current batch input
    float *d_target;       // Current batch target

} NeuralNetwork;

/**
 * Initialize neural network on GPU
 */
NeuralNetwork* create_network_gpu(
    int input_size,
    int hidden_size,
    int output_size,
    int batch_size
) {
    NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;
    nn->batch_size = batch_size;
    nn->timestep = 0;

    // Allocate device memory for layer 1
    cudaMalloc(&nn->d_W1, input_size * hidden_size * sizeof(float));
    cudaMalloc(&nn->d_b1, hidden_size * sizeof(float));
    cudaMalloc(&nn->d_z1, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&nn->d_a1, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&nn->d_grad_W1, input_size * hidden_size * sizeof(float));
    cudaMalloc(&nn->d_grad_b1, hidden_size * sizeof(float));
    cudaMalloc(&nn->d_grad_z1, batch_size * hidden_size * sizeof(float));

    // Allocate device memory for layer 2
    cudaMalloc(&nn->d_W2, hidden_size * output_size * sizeof(float));
    cudaMalloc(&nn->d_b2, output_size * sizeof(float));
    cudaMalloc(&nn->d_z2, batch_size * output_size * sizeof(float));
    cudaMalloc(&nn->d_a2, batch_size * output_size * sizeof(float));
    cudaMalloc(&nn->d_grad_W2, hidden_size * output_size * sizeof(float));
    cudaMalloc(&nn->d_grad_b2, output_size * sizeof(float));
    cudaMalloc(&nn->d_grad_z2, batch_size * output_size * sizeof(float));

    // Allocate optimizer state (Adam)
    cudaMalloc(&nn->d_m_W1, input_size * hidden_size * sizeof(float));
    cudaMalloc(&nn->d_v_W1, input_size * hidden_size * sizeof(float));
    cudaMalloc(&nn->d_m_b1, hidden_size * sizeof(float));
    cudaMalloc(&nn->d_v_b1, hidden_size * sizeof(float));
    cudaMalloc(&nn->d_m_W2, hidden_size * output_size * sizeof(float));
    cudaMalloc(&nn->d_v_W2, hidden_size * output_size * sizeof(float));
    cudaMalloc(&nn->d_m_b2, output_size * sizeof(float));
    cudaMalloc(&nn->d_v_b2, output_size * sizeof(float));

    // Initialize optimizer state to zero
    cudaMemset(nn->d_m_W1, 0, input_size * hidden_size * sizeof(float));
    cudaMemset(nn->d_v_W1, 0, input_size * hidden_size * sizeof(float));
    cudaMemset(nn->d_m_b1, 0, hidden_size * sizeof(float));
    cudaMemset(nn->d_v_b1, 0, hidden_size * sizeof(float));
    cudaMemset(nn->d_m_W2, 0, hidden_size * output_size * sizeof(float));
    cudaMemset(nn->d_v_W2, 0, hidden_size * output_size * sizeof(float));
    cudaMemset(nn->d_m_b2, 0, output_size * sizeof(float));
    cudaMemset(nn->d_v_b2, 0, output_size * sizeof(float));

    // Allocate input/output buffers
    cudaMalloc(&nn->d_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&nn->d_target, batch_size * output_size * sizeof(float));

    // Initialize weights (Xavier initialization)
    // TODO: Implement weight initialization kernel

    return nn;
}

/**
 * Forward pass through network
 */
void forward_pass_gpu(NeuralNetwork *nn) {
    // Layer 1: z1 = input @ W1 + b1
    matmul_tiled_gpu(  // From Phase 4
        nn->d_input, nn->d_W1, nn->d_z1,
        nn->batch_size, nn->input_size, nn->hidden_size
    );
    add_bias_gpu(nn->d_z1, nn->d_b1, nn->batch_size, nn->hidden_size);

    // Layer 1: a1 = ReLU(z1)
    relu_forward_gpu(nn->d_z1, nn->d_a1, nn->batch_size * nn->hidden_size);

    // Layer 2: z2 = a1 @ W2 + b2
    matmul_tiled_gpu(
        nn->d_a1, nn->d_W2, nn->d_z2,
        nn->batch_size, nn->hidden_size, nn->output_size
    );
    add_bias_gpu(nn->d_z2, nn->d_b2, nn->batch_size, nn->output_size);

    // Layer 2: a2 = Softmax(z2)
    softmax_forward_gpu(nn->d_z2, nn->d_a2, nn->batch_size, nn->output_size);
}

/**
 * Backward pass (backpropagation)
 */
void backward_pass_gpu(NeuralNetwork *nn) {
    // Output layer gradient: d_z2 = a2 - target
    softmax_crossentropy_backward_gpu(
        nn->d_a2, nn->d_target, nn->d_grad_z2,
        nn->batch_size, nn->output_size
    );

    // Layer 2 gradients
    dense_layer_backward_gpu(
        nn->d_a1,          // input to layer 2
        nn->d_grad_z2,     // gradient from loss
        nn->d_W2,          // weights
        nn->d_grad_W2,     // output: weight gradients
        nn->d_grad_b2,     // output: bias gradients
        nn->d_grad_z1,     // output: gradient to layer 1
        nn->batch_size,
        nn->hidden_size,
        nn->output_size,
        true              // use tiled version
    );

    // ReLU backward
    relu_backward_gpu(
        nn->d_grad_z1, nn->d_z1, nn->d_grad_z1,
        nn->batch_size * nn->hidden_size
    );

    // Layer 1 gradients
    float *d_grad_input;  // We don't need this, but API requires it
    cudaMalloc(&d_grad_input, nn->batch_size * nn->input_size * sizeof(float));

    dense_layer_backward_gpu(
        nn->d_input,
        nn->d_grad_z1,
        nn->d_W1,
        nn->d_grad_W1,
        nn->d_grad_b1,
        d_grad_input,
        nn->batch_size,
        nn->input_size,
        nn->hidden_size,
        true
    );

    cudaFree(d_grad_input);
}

/**
 * Update weights using Adam optimizer
 */
void update_weights_gpu(
    NeuralNetwork *nn,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon
) {
    nn->timestep++;

    // Update layer 1 weights and biases
    adam_update_gpu(
        nn->d_W1, nn->d_grad_W1, nn->d_m_W1, nn->d_v_W1,
        learning_rate, beta1, beta2, epsilon, nn->timestep,
        nn->input_size * nn->hidden_size
    );
    adam_update_gpu(
        nn->d_b1, nn->d_grad_b1, nn->d_m_b1, nn->d_v_b1,
        learning_rate, beta1, beta2, epsilon, nn->timestep,
        nn->hidden_size
    );

    // Update layer 2 weights and biases
    adam_update_gpu(
        nn->d_W2, nn->d_grad_W2, nn->d_m_W2, nn->d_v_W2,
        learning_rate, beta1, beta2, epsilon, nn->timestep,
        nn->hidden_size * nn->output_size
    );
    adam_update_gpu(
        nn->d_b2, nn->d_grad_b2, nn->d_m_b2, nn->d_v_b2,
        learning_rate, beta1, beta2, epsilon, nn->timestep,
        nn->output_size
    );
}

/**
 * Train one batch
 */
float train_batch_gpu(
    NeuralNetwork *nn,
    const float *h_input,
    const float *h_target,
    float learning_rate
) {
    // Copy data to GPU
    cudaMemcpy(nn->d_input, h_input,
               nn->batch_size * nn->input_size * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(nn->d_target, h_target,
               nn->batch_size * nn->output_size * sizeof(float),
               cudaMemcpyHostToDevice);

    // Forward pass
    forward_pass_gpu(nn);

    // Compute loss
    float loss = cross_entropy_loss_gpu(
        nn->d_a2, nn->d_target, nn->batch_size, nn->output_size
    );

    // Backward pass
    backward_pass_gpu(nn);

    // Update weights (Adam optimizer)
    update_weights_gpu(nn, learning_rate, 0.9f, 0.999f, 1e-8f);

    return loss;
}
```

---

## 🚀 Exercise 5.6: Python Integration

### `python/cuda_nn/neural_network.py`

```python
"""
Python wrapper for GPU-accelerated neural network
"""

import ctypes
import numpy as np
from typing import Tuple, Optional

# Load shared library
lib = ctypes.CDLL('./libnn_gpu.so')  # or .dll on Windows

# Define C structures and function signatures
# ... (similar to Phase 3 ctypes example)

class NeuralNetworkGPU:
    """
    GPU-accelerated neural network

    Example:
        nn = NeuralNetworkGPU(784, 128, 10, batch_size=32)
        nn.fit(X_train, y_train, epochs=10, learning_rate=0.001)
        accuracy = nn.evaluate(X_test, y_test)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        batch_size: int = 32
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        # Create network on GPU
        # self._nn = lib.create_network_gpu(...)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        learning_rate: float = 0.001,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True
    ):
        """
        Train the network

        Args:
            X: Input data (n_samples, input_size)
            y: Labels (n_samples,) - will be one-hot encoded
            epochs: Number of training epochs
            learning_rate: Learning rate
            validation_data: Optional (X_val, y_val) for validation
            verbose: Print progress
        """
        n_samples = X.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        # One-hot encode labels
        y_one_hot = self._one_hot_encode(y, self.output_size)

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_one_hot[indices]

            total_loss = 0.0

            for batch_idx in range(n_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, n_samples)

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Pad last batch if necessary
                if X_batch.shape[0] < self.batch_size:
                    pad_size = self.batch_size - X_batch.shape[0]
                    X_batch = np.vstack([X_batch, np.zeros((pad_size, self.input_size))])
                    y_batch = np.vstack([y_batch, np.zeros((pad_size, self.output_size))])

                # Train batch on GPU
                loss = lib.train_batch_gpu(
                    self._nn,
                    X_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    y_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_float(learning_rate)
                )

                total_loss += loss

            avg_loss = total_loss / n_batches

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}", end="")

                if validation_data is not None:
                    val_acc = self.evaluate(*validation_data)
                    print(f" - Val Acc: {val_acc:.4f}")
                else:
                    print()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        # Run forward pass on GPU
        # Return probabilities
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy on test data"""
        predictions = self.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = y if y.ndim == 1 else np.argmax(y, axis=1)
        accuracy = np.mean(predicted_classes == true_classes)
        return accuracy

    def _one_hot_encode(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        """Convert integer labels to one-hot encoding"""
        n = labels.shape[0]
        one_hot = np.zeros((n, num_classes), dtype=np.float32)
        one_hot[np.arange(n), labels] = 1
        return one_hot

    def __del__(self):
        """Free GPU memory"""
        if hasattr(self, '_nn'):
            lib.free_network_gpu(self._nn)
```

---

## 🎯 Final Project: MNIST Digit Classifier

### `python/examples/mnist_gpu.py`

```python
"""
MNIST Digit Classification on GPU

Goal: >97% accuracy, <30 seconds training time
"""

import numpy as np
import time
from cuda_nn import NeuralNetworkGPU

# Load MNIST (use keras datasets or manual download)
from keras.datasets import mnist

def load_and_preprocess_mnist():
    """Load and normalize MNIST data"""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshape and normalize
    X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 784).astype('float32') / 255.0

    return (X_train, y_train), (X_test, y_test)

def main():
    print("="*70)
    print(" MNIST Digit Classification on GPU")
    print("="*70)

    # Load data
    print("\nLoading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_mnist()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # Create network
    print("\nCreating neural network on GPU...")
    nn = NeuralNetworkGPU(
        input_size=784,
        hidden_size=128,
        output_size=10,
        batch_size=64
    )

    # Train
    print("\nTraining...\n")
    start_time = time.time()

    nn.fit(
        X_train, y_train,
        epochs=10,
        learning_rate=0.001,
        validation_data=(X_test, y_test),
        verbose=True
    )

    training_time = time.time() - start_time

    # Evaluate
    print("\nEvaluating...")
    test_accuracy = nn.evaluate(X_test, y_test)

    # Results
    print("\n" + "="*70)
    print(" RESULTS")
    print("="*70)
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    print(f"Target: >97% accuracy, <30 seconds")

    if test_accuracy > 0.97 and training_time < 30:
        print("\n🎉 SUCCESS! You've achieved the goal!")
    elif test_accuracy > 0.97:
        print("\n✓ Great accuracy! Try optimizing for speed.")
    elif training_time < 30:
        print("\n✓ Fast training! Try improving accuracy.")
    else:
        print("\n Keep optimizing - you'll get there!")

    print("="*70)

if __name__ == "__main__":
    main()
```

---

## ✅ Success Criteria

- [ ] All activation functions work on GPU
- [ ] Loss computation on GPU
- [ ] Backward pass computes correct gradients
- [ ] SGD optimizer updates weights
- [ ] Adam optimizer works (bonus)
- [ ] Complete training loop executes
- [ ] Python API is clean and easy to use
- [ ] MNIST classifier achieves >95% accuracy
- [ ] **Stretch goal**: >97% accuracy in <30 seconds

---

## 🎓 Understanding Questions

### 1. Memory Management
How do you avoid memory leaks when allocating temporary GPU memory in each kernel?

### 2. Backward Pass
Why does the softmax + cross-entropy combination simplify the gradient to `pred - target`?

### 3. Optimization
Why is Adam generally better than SGD for deep learning?

### 4. Performance
What's the bottleneck in your implementation? Memory or computation?

---

## 📊 Expected Performance

### Baseline (CPU NumPy):
- Training time: ~5-10 minutes
- Accuracy: ~97%

### Your GPU Implementation:
- Training time: **<30 seconds** (10-20x faster!)
- Accuracy: **>97%**

### Professional libraries (PyTorch/TensorFlow):
- Training time: ~10-20 seconds (with cuDNN)
- Accuracy: ~98-99% (with better architectures)

**You'll be very close to professional performance!**

---

## 🚀 Optimization Tips

1. **Use shared memory** in all kernels where possible
2. **Minimize host-device transfers** - keep data on GPU
3. **Batch operations** - don't transfer after each layer
4. **Profile with nvprof** - find actual bottlenecks
5. **Try larger batch sizes** - better GPU utilization
6. **Experiment with architectures** - more/fewer hidden units

---

## 🎯 Next Steps After Completion

Once you finish this phase, you'll have:
- ✅ Complete understanding of neural networks
- ✅ GPU programming skills
- ✅ C and Python integration expertise
- ✅ Working MNIST classifier on GPU

**Possible extensions**:
1. Add convolutional layers (CNNs)
2. Implement batch normalization
3. Try CIFAR-10 dataset (color images)
4. Multi-GPU training
5. Compare with PyTorch/TensorFlow

---

## 📚 Resources

- **Backpropagation**: CS231n notes on backprop
- **Adam optimizer**: Original paper by Kingma & Ba
- **GPU optimization**: NVIDIA CUDA Best Practices Guide
- **Numerical stability**: Why softmax subtracts max

---

**Start with Exercise 5.1 (Activation Functions)! Build up layer by layer.**

**Show me your code as you progress! This is the most complex phase - we'll work through it together!** 🚀
