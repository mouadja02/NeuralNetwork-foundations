"""
Python bindings for C matrix library using ctypes.

This is the SIMPLEST way to call C from Python.
No compilation needed, just load the shared library!
"""

import ctypes
import os
import sys
from typing import List

# Determine library name based on platform
if sys.platform == 'win32':
    lib_name = 'libmatrix.dll'
elif sys.platform == 'darwin':
    lib_name = 'libmatrix.dylib'
else:
    lib_name = 'libmatrix.so'

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), '..', '..', 'c', lib_name)
lib = ctypes.CDLL(lib_path)


# Define Matrix structure (must match C struct exactly!)
class CMatrix(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('rows', ctypes.c_size_t),
        ('cols', ctypes.c_size_t)
    ]


# Define function signatures
# matrix_create(rows, cols) -> Matrix*
lib.matrix_create.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
lib.matrix_create.restype = ctypes.POINTER(CMatrix)

# matrix_free(Matrix*)
lib.matrix_free.argtypes = [ctypes.POINTER(CMatrix)]
lib.matrix_free.restype = None

# matrix_add(Matrix*, Matrix*) -> Matrix*
lib.matrix_add.argtypes = [ctypes.POINTER(CMatrix), ctypes.POINTER(CMatrix)]
lib.matrix_add.restype = ctypes.POINTER(CMatrix)

# matrix_multiply(Matrix*, Matrix*) -> Matrix*
lib.matrix_multiply.argtypes = [ctypes.POINTER(CMatrix), ctypes.POINTER(CMatrix)]
lib.matrix_multiply.restype = ctypes.POINTER(CMatrix)

# matrix_transpose(Matrix*) -> Matrix*
lib.matrix_transpose.argtypes = [ctypes.POINTER(CMatrix)]
lib.matrix_transpose.restype = ctypes.POINTER(CMatrix)


class Matrix:
    """Python wrapper for C Matrix"""

    def __init__(self, data: List[List[float]]):
        """Create matrix from Python list"""
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0

        # Create C matrix
        self._c_matrix = lib.matrix_create(self.rows, self.cols)

        # Copy data
        flat_data = [item for row in data for item in row]
        for i, val in enumerate(flat_data):
            self._c_matrix.contents.data[i] = val

    def __del__(self):
        """Free C memory when Python object is deleted"""
        if hasattr(self, '_c_matrix') and self._c_matrix:
            lib.matrix_free(self._c_matrix)

    def to_list(self) -> List[List[float]]:
        """Convert C matrix back to Python list"""
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                idx = i * self.cols + j
                row.append(self._c_matrix.contents.data[idx])
            result.append(row)
        return result

    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Matrix addition using C"""
        result_c = lib.matrix_add(self._c_matrix, other._c_matrix)
        return Matrix._from_c_matrix(result_c)

    def dot(self, other: 'Matrix') -> 'Matrix':
        """Matrix multiplication using C"""
        result_c = lib.matrix_multiply(self._c_matrix, other._c_matrix)
        return Matrix._from_c_matrix(result_c)

    def T(self) -> 'Matrix':
        """Transpose using C"""
        result_c = lib.matrix_transpose(self._c_matrix)
        return Matrix._from_c_matrix(result_c)

    @classmethod
    def _from_c_matrix(cls, c_matrix):
        """Create Python Matrix from existing C Matrix"""
        rows = c_matrix.contents.rows
        cols = c_matrix.contents.cols

        # Extract data
        data = []
        for i in range(rows):
            row = []
            for j in range(cols):
                idx = i * cols + j
                row.append(c_matrix.contents.data[idx])
            data.append(row)

        # Create Python matrix (this will allocate new C memory)
        result = cls(data)

        # Free the temporary C matrix
        lib.matrix_free(c_matrix)

        return result


# Test it!
if __name__ == "__main__":
    print("Testing ctypes bindings...")

    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])

    c = a + b
    print("Addition:", c.to_list())

    d = a.dot(b)
    print("Multiplication:", d.to_list())

    e = a.T()
    print("Transpose:", e.to_list())