import numpy as np
import sympy as sp

# Creating the first matrix
matrixA = np.random.randint(1, 9, (10, 10))
matrixB = np.random.randint(1, 9, (10, 10))

print("\n\nFirst matrix 10x10:\n\n")
sp.pprint(sp.Matrix(matrixA))
print("\n\n\n")

print("Second matrix 10x10\n\n")
sp.pprint(sp.Matrix(matrixB))
print("\n\n\n")

# Modifying the fist matrix by adding a '1' diagonal
np.fill_diagonal(matrixA, 1)
print("Modified matrix 1 10x10:\n\n")
sp.pprint(sp.Matrix(matrixA))
print("\n\n\n")

# Modifying the second matrix by adding a '2' diagonal
np.fill_diagonal(matrixB, 2)
print("Modified matrix 2 10x10:\n\n")
sp.pprint(sp.Matrix(matrixB))
print("\n\n\n")

# Adding matrixA and matrixB
print("Sum of matrix A and B 10x10:\n\n")
sp.pprint(sp.Matrix(matrixA + matrixB))
print("\n\n\n")

# Diff matrixA and matrixB
print("Diff between matrix A and B 10x10:\n\n")
sp.pprint(sp.Matrix(matrixA - matrixB))
print("\n\n\n")

# Mult matrixA and matrixB
print("Multiplying matrix A and B 10x10:\n\n")
sp.pprint(sp.Matrix(matrixA * matrixB))
print("\n\n\n")

# Transpose of matrixA
print("Transpose of matrix A 10x10:\n\n")
sp.pprint(sp.Matrix(matrixB.transpose()))
print("\n\n\n")

# Transpose of matrixB
print("Transpose of matrix B 10x10:\n\n")
sp.pprint(sp.Matrix(matrixB.transpose()))
print("\n\n\n")

# Creating vectors
vectorC = np.random.randint(1000,32435, 3)
vectorD = np.random.randint(1000,32435, 3)

print("First vector:\n\n")
sp.pprint(sp.Matrix(vectorC))
print("\n\n\n")
print("Second vector:\n\n ")
sp.pprint(sp.Matrix(vectorD))
print("\n\n\n")

# Dot prod. VectorC and vectorD
dotProduct = np.dot(vectorC, vectorD)
print("Dot prod. between Vector C and vector D :\n\n %s \n\n\n\n" % str(dotProduct))


# Cross prod. vectorC and vectorD
crossProduct = np.cross(vectorC, vectorD, 0)
print("Cross prod. between vector C and vector D:\n\n")
sp.pprint(sp.Matrix(crossProduct))
print("\n\n\n")

# Creating Matrix
notFerr = sp.Matrix([[3, -6, 9, 0, -3, 18], 
                     [-1, 2, -3, 2, 11, 2], 
                     [2, -4, 6, -2, -12, 4]])
print("Non-RREF matrix:\n\n" )
sp.pprint(notFerr)
print("\n\n\n\n")


# RREF
print("RREF matrix:\n\n") 
sp.pprint(notFerr.rref(simplify=True, pivots=False,normalize_last=True))