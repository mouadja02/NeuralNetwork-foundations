"""
Matrix operations from scratch - No NumPy!

This is your implementation. Fill in the TODO sections.
"""


class Matrix:
    """
    A simple matrix implementation using nested Python lists.
    """

    def __init__(self, data):
        """Initialize matrix from 2D list."""
        # 1. Validate that all rows have the same length
        for row in data:
            if len(row) != len(data[0]):
                raise ValueError("All rows must have the same length.")
        # 2. Store the data
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0

    def shape(self):
        """Return tuple (rows, cols)"""
        return (self.rows, self.cols)

    def __add__(self, other):
        """Add two matrices element-wise."""
        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same dimensions to add.")
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j]+ other.data[i][j])
            result.append(row)
        return Matrix(result)


    def __sub__(self, other):
        """Add two matrices element-wise."""
        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same dimensions to add.")
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j] - other.data[i][j])
            result.append(row)
        return Matrix(result)
    
    def __mul__(self, other):
        """Element-wise multiplication (Hadamard product)."""
        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same dimensions to multiply element-wise.")
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(self.data[i][j]* other.data[i][j])
            result.append(row)
        return Matrix(result)

    def dot(self, other):
        """Matrix multiplication."""
        if self.cols != other.rows:
            raise ValueError("Incompatible dimensions for matrix multiplication.")
        result = []
        for i in range(self.rows):
            row = []
            for j in range(other.cols):
                sum = 0
                for k in range(self.cols):
                    sum += self.data[i][k]*other.data[k][j]
                row.append(sum)
            result.append(row)
        return Matrix(result)

    def T(self):
        """Transpose matrix."""
        result = []
        for i in range(self.cols):
            row = []
            for j in range(self.rows):
                row.append(self.data[j][i])
            result.append(row)
        return Matrix(result)

    def apply(self, func):
        """Apply function to every element."""
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(func(self.data[i][j]))
            result.append(row)
        return Matrix(result)

    def __repr__(self, print_2d=False):
        """Display matrix in readable format."""
        if print_2d:
            print("\nMatrix:")
            for row in self.data:
                print("  ", " ".join(f"{x:6.2f}" for x in row))
            print()
        return f"Matrix({self.data})"       

    def __eq__(self, other):
        """Check equality."""
        if self.shape() != other.shape():
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if self.data[i][j] != other.data[i][j]:
                    return False
        return True


def zeros(rows, cols):
    """Create matrix of zeros."""
    return Matrix([[0 for _ in range(cols)] for _ in range(rows)])


def ones(rows, cols):
    """Create matrix of ones."""
    return Matrix([[1 for _ in range(cols)] for _ in range(rows)])


def identity(n):
    """Create identity matrix."""
    return Matrix([[1 if i==j else 0 for i in range(n)] for j in range(n)])
