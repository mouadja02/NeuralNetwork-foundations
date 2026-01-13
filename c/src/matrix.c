#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "matrix.h"

/**
 * Create a new matrix with uninitialized data
 */
Matrix* matrix_create(size_t rows, size_t cols) {
  if (rows == 0 || cols == 0){
    fprintf(stderr, "Error: Cannot create matrix with 0 dimensions\n");
    return NULL;
  }
  Matrix *m = (Matrix*)malloc(sizeof(Matrix));
  if (!m){
    fprintf(stderr, "Error: Failed to allocate matrix structure\n");
    return NULL;
  }
  m->rows = rows;
  m->cols = cols;
  m->data = (float*)malloc(rows * cols * sizeof(float));
  if (!m->data){
    fprintf(stderr, "Error: Failed to allocate matrix data\n");
    free(m);
    return NULL;
  }
  return m;
}

 
/**
 * Create matrix from existing array (copies data)
 */
Matrix* matrix_create_from_array(float *data, size_t rows, size_t cols) {
  Matrix *m = matrix_create(rows, cols);
  if (!m) return NULL;
  memcpy(m->data, data, rows * cols * sizeof(float));
  return m;
}

/**
 * Free matrix memory
 * IMPORTANT: Always call this when done with a matrix!
*/
void matrix_free(Matrix *m) {
  if (m){
    if (m->data){
      free(m->data);
    }
    free(m);
  }
}

/**
 * Add two matrices element-wise
 */
Matrix* matrix_add(const Matrix *a, const Matrix *b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    fprintf(stderr, "Error: Matrix dimensions must match for addition\n");
    return NULL;
  }
  Matrix *result = matrix_create(a->rows, a->cols);
  if (!result) return NULL;
  size_t size = a->rows * a->cols;
  for (size_t i=0; i<size; i++) {
    result->data[i] = a->data[i] + b->data[i];
  }
  return result;
}

/**
 * Subtract two matrices element-wise
 */
 Matrix* matrix_subtract(const Matrix *a, const Matrix *b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    fprintf(stderr, "Error: Matrix dimensions must match for subtraction\n");
    return NULL;
  }
  Matrix *result = matrix_create(a->rows, a->cols);
  if (!result) return NULL;
  size_t size = a->rows * a->cols;
  for (size_t i=0; i<size; i++) {
    result->data[i] = a->data[i] - b->data[i];
  }
  return result;
}


/**
 * Multiply two matrices element-wise (Hadamard product)
 */
Matrix* matrix_multiply_elementwise(const Matrix *a, const Matrix *b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    fprintf(stderr, "Error: Matrix dimensions must match for element-wise multiplication\n");
    return NULL;
  }
  Matrix *result = matrix_create(a->rows, a->cols);
  if (!result) return NULL;
  size_t size = a->rows * a->cols;
  for (size_t i=0; i<size; i++) {
    result->data[i] = a->data[i] * b->data[i];
  }
  return result;
}



/**
 * Matrix multiplication: C = A × B
 *
 * A: (m × n)
 * B: (n × p)
 * C: (m × p)
 *
 * C[i,j] = sum(A[i,k] * B[k,j]) for k=0 to n-1
 */
Matrix* matrix_multiply(const Matrix *a, const Matrix *b) {
  if (a->cols != b->rows) {
    fprintf(stderr, "Error: Matrix dimensions incompatible for multiplication\n");
    fprintf(stderr, "  A: %zu×%zu, B: %zu×%zu\n", a->rows, a->cols, b->rows, b->cols);
    return NULL;
  }
  Matrix *result = matrix_create(a->rows, b->cols);
  if (!result) return NULL;
  memset(result->data, 0, a->rows * b->cols * sizeof(float));
  for (size_t i=0; i<a->rows; i++) {
    for (size_t j=0; j<b->cols; j++) {
      float sum = 0.0f;
      for (size_t k=0; k<a->cols; k++) {
        sum += a->data[i * a-> cols + k] * b->data[k * b->cols + j];
      }
      result -> data[i * b->cols + j] = sum;
    }
  }
  return result;
}



/**
 * Transpose matrix
 */
Matrix* matrix_transpose(const Matrix *m) {
  Matrix *result = matrix_create(m->cols, m->rows);
  if (!result) return NULL;
  for (size_t i=0; i<m->rows; i++) {
    for (size_t j=0; j<m->cols; j++) {
      result->data[j * m->rows + i] = m->data[i * m->cols + j];
    }
  }
  return result;
}

/**
 * Print matrix to stdout
 */
void matrix_print(const Matrix *m) {
  printf("Matrix (%zu×%zu):\n", m->rows, m->cols);
  for (size_t i=0; i<m->rows; i++) {
    printf("  [");
    for (size_t j=0; j<m->cols; j++) {
      printf("%8.4f", m->data[i * m->cols + j]);
    }
  }
  printf("]\n");
}

/**
 * Fill matrix with a constant value
 */
void matrix_fill(Matrix *m, float value) {
  size_t size = m->rows * m->cols;
  for (size_t i=0; i<size; i++) {
    m->data[i] = value;
  }
}


/**
 * Fill matrix with random values in [min, max]
 */
void matrix_randomize(Matrix *m, float min, float max) {
  static int seeded = 0;
  if (!seeded) {
    srand(time(NULL));
    seeded = 1;
  }
  size_t size = m->rows * m->cols;
  float range = max - min;
  for (size_t i=0; i<size; i++) {
    m->data[i] = min + ((float)rand() / RAND_MAX) * range;
  }
}

/**
 * Check if two matrices are equal within epsilon
 */
int matrix_equal(const Matrix *a, const Matrix *b, float epsilon) {
  if (a->rows != b->rows || a->cols != b->cols) {
    return 0;
  }
  size_t size = a->rows * a->cols;
  for (size_t i=0; i<size; i++) {
    if (fabsf(a->data[i] - b->data[i]) > epsilon) {
      return 0;
    }
  }
  return 1;
}

/**
 * Create zero matrix
 */
Matrix* matrix_zeros(size_t rows, size_t cols) {
  Matrix *m = matrix_create(rows, cols);
  if (m) {
    matrix_fill(m, 0.0f);
  }
  return m;
}

/**
 * Create matrix of ones
 */
Matrix* matrix_ones(size_t rows, size_t cols) {
  Matrix *m = matrix_create(rows, cols);
  if (m) {
    matrix_fill(m, 1.0f);
  }
  return m;
}

/**
 * Create identity matrix
 */
Matrix* matrix_identity(size_t n) {
  Matrix *m = matrix_zeros(n, n);
  if (m) {
    for (size_t i=0; i<n; i++) {
      m->data[i * n + i] = 1.0f;
    }
  }
  return m;
}