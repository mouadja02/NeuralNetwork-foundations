#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

extern "C" {
    #include "matrix.h"
}

namespace py = pybind11;

class MatrixWrapper {
private:
    Matrix* m;

public:
    MatrixWrapper(const std::vector<std::vector<float>>& data) {
        size_t rows = data.size();
        size_t cols = data[0].size();

        m = matrix_create(rows, cols);

        // Copy data
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                m->data[i * cols + j] = data[i][j];
            }
        }
    }

    ~MatrixWrapper() {
        matrix_free(m);
    }

    std::vector<std::vector<float>> to_list() const {
        std::vector<std::vector<float>> result(m->rows);
        for (size_t i = 0; i < m->rows; i++) {
            result[i].resize(m->cols);
            for (size_t j = 0; j < m->cols; j++) {
                result[i][j] = m->data[i * m->cols + j];
            }
        }
        return result;
    }

    MatrixWrapper add(const MatrixWrapper& other) const {
        Matrix* result = matrix_add(m, other.m);
        return MatrixWrapper(result);
    }

    MatrixWrapper dot(const MatrixWrapper& other) const {
        Matrix* result = matrix_multiply(m, other.m);
        return MatrixWrapper(result);
    }

    MatrixWrapper transpose() const {
        Matrix* result = matrix_transpose(m);
        return MatrixWrapper(result);
    }

private:
    // Private constructor for wrapping existing Matrix*
    MatrixWrapper(Matrix* mat) : m(mat) {}
};

PYBIND11_MODULE(cmatrix, m) {
    m.doc() = "C matrix library with pybind11 bindings";

    py::class_<MatrixWrapper>(m, "Matrix")
        .def(py::init<const std::vector<std::vector<float>>&>())
        .def("to_list", &MatrixWrapper::to_list)
        .def("add", &MatrixWrapper::add)
        .def("dot", &MatrixWrapper::dot)
        .def("transpose", &MatrixWrapper::transpose)
        .def("__add__", &MatrixWrapper::add)
        .def("T", &MatrixWrapper::transpose);
}
