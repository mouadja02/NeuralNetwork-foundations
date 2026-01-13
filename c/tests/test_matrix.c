#include <stdio.h>
#include <assert.h>
#include "../include/matrix.h"

void test_matrix_creation() {
    printf("Testing matrix creation... ");

    Matrix *m = matrix_create(3, 4);
    assert(m != NULL);
    assert(m->rows == 3);
    assert(m->cols == 4);

    matrix_free(m);
    printf("✓ PASSED\n");
}

void test_matrix_addition() {
    printf("Testing matrix addition... ");

    float data_a[] = {1, 2, 3, 4};
    float data_b[] = {5, 6, 7, 8};

    Matrix *a = matrix_create_from_array(data_a, 2, 2);
    Matrix *b = matrix_create_from_array(data_b, 2, 2);
    Matrix *c = matrix_add(a, b);

    assert(c->data[0] == 6);  // 1 + 5
    assert(c->data[1] == 8);  // 2 + 6
    assert(c->data[2] == 10); // 3 + 7
    assert(c->data[3] == 12); // 4 + 8

    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
    printf("✓ PASSED\n");
}

void test_matrix_multiplication() {
    printf("Testing matrix multiplication... ");

    // [[1, 2],     [[5, 6],     [[19, 22],
    //  [3, 4]]  ×   [7, 8]]  =   [43, 50]]

    float data_a[] = {1, 2, 3, 4};
    float data_b[] = {5, 6, 7, 8};

    Matrix *a = matrix_create_from_array(data_a, 2, 2);
    Matrix *b = matrix_create_from_array(data_b, 2, 2);
    Matrix *c = matrix_multiply(a, b);

    assert(c->data[0] == 19); // 1*5 + 2*7 = 19
    assert(c->data[1] == 22); // 1*6 + 2*8 = 22
    assert(c->data[2] == 43); // 3*5 + 4*7 = 43
    assert(c->data[3] == 50); // 3*6 + 4*8 = 50

    matrix_free(a);
    matrix_free(b);
    matrix_free(c);
    printf("✓ PASSED\n");
}

void test_matrix_transpose() {
    printf("Testing matrix transpose... ");

    float data[] = {1, 2, 3, 4, 5, 6};
    Matrix *a = matrix_create_from_array(data, 2, 3);
    Matrix *b = matrix_transpose(a);

    assert(b->rows == 3);
    assert(b->cols == 2);
    assert(b->data[0] == 1);
    assert(b->data[1] == 4);
    assert(b->data[2] == 2);
    assert(b->data[3] == 5);

    matrix_free(a);
    matrix_free(b);
    printf("✓ PASSED\n");
}

int main() {
    printf("\n==================================================\n");
    printf(" 🧪 TESTING C MATRIX IMPLEMENTATION\n");
    printf("==================================================\n\n");

    test_matrix_creation();
    test_matrix_addition();
    test_matrix_multiplication();
    test_matrix_transpose();

    printf("\n==================================================\n");
    printf("🎉 ALL C TESTS PASSED!\n");
    printf("==================================================\n\n");

    return 0;
}