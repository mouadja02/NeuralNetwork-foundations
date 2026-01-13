#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>

/**
 * Matrix structure
 *
 * Data is stored in row-major order:
 * For a 2x3 matrix: [a, b, c, d, e, f]
 * Represents: [[a, b, c],
 *              [d, e, f]]
 *
 * Access element (i,j): data[i * cols + j]
 */
typedef struct {
    float *data;    // Pointer to dynamically allocated array
    size_t rows;    // Number of rows
    size_t cols;    // Number of columns
} Matrix;

// Creation & Destruction
Matrix* matrix_create(size_t rows, size_t cols);
Matrix* matrix_create_from_array(float *data, size_t rows, size_t cols);
void matrix_free(Matrix *m);

// Basic Operations
Matrix* matrix_add(const Matrix *a, const Matrix *b);
Matrix* matrix_subtract(const Matrix *a, const Matrix *b);
Matrix* matrix_multiply_elementwise(const Matrix *a, const Matrix *b);
Matrix* matrix_multiply(const Matrix *a, const Matrix *b);
Matrix* matrix_transpose(const Matrix *m);
// Utility
void matrix_print(const Matrix *m);
void matrix_fill(Matrix *m, float value);
void matrix_randomize(Matrix *m, float min, float max);
int matrix_equal(const Matrix *a, const Matrix *b, float epsilon);

// Special Matrices
Matrix* matrix_zeros(size_t rows, size_t cols);
Matrix* matrix_ones(size_t rows, size_t cols);
Matrix* matrix_identity(size_t n);

#endif // MATRIX_H